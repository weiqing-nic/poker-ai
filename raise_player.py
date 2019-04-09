import numpy as np
from keras.layers import Input, Dense, Conv2D, concatenate, Flatten
from keras.models import Model
from pypokerengine.players import BasePokerPlayer
from pypokerengine.engine.card import Card


class Group18Player(BasePokerPlayer):
    suits = [Card.SUIT_MAP[key] for key in Card.SUIT_MAP]
    ranks = [Card.RANK_MAP[key] for key in Card.RANK_MAP]
    max_no_of_rounds = 4
    target_Q = np.zeros((1, 3), dtype=float)
    old_small_blind_cards = np.zeros((1, len(suits), len(ranks), max_no_of_rounds), dtype=float)
    old_small_blind_actions = np.zeros((1, 2, 6, 4), dtype=float)
    old_small_blind_position = np.zeros(1, dtype=float)
    old_small_blind_features = [old_small_blind_cards, old_small_blind_actions, old_small_blind_position]

    def __init__(self):
        super(Group18Player, self).__init__()

        self.vvh = 0

        def keras_model():

            input_cards = Input(shape=(len(Group18Player.suits),
                                       len(Group18Player.ranks),
                                       Group18Player.max_no_of_rounds),
                                name="cards_input")
            input_actions = Input(shape=(2, 6, 4), name="actions_input")
            input_position = Input(shape=(1,), name="position_input")

            x1 = Conv2D(32, (2, 2), activation='relu')(input_cards)
            x2 = Conv2D(32, (2, 2), activation='relu')(input_actions)
            x3 = Dense(1, activation='relu')(input_position)

            d1 = Dense(128, activation='relu')(x1)
            d1 = Flatten()(d1)
            d2 = Dense(128, activation='relu')(x2)
            d2 = Flatten()(d2)
            x = concatenate([d1, d2, x3])
            x = Dense(128)(x)
            x = Dense(32)(x)
            out = Dense(3)(x)

            model = Model(inputs=[input_cards, input_actions, input_position],
                          outputs=out)
            if self.vvh == 0:
                model.load_weights('./setup/weights.h5', by_name=True)
            model.compile(optimizer='rmsprop', loss='mse')

            return model

        self.table = {}
        self.initial_stack = 0
        self.experience_state = []
        self.experience_reward = []
        self.has_played = False
        self.model = keras_model()
        self.all_Q_small_blind = np.array([[0.0, 0.0]], dtype=float)
        self.small_blind_features = np.array([0.0, 0.0, 0.0], dtype=float)
        self.small_blind_position = -1

    def declare_action(self, valid_actions, hole_card, round_state):

        def get_suit_index(card):
            return Group18Player.suits.index(card[0])

        def get_rank_index(card):
            return Group18Player.ranks.index(card[1])

        def get_street_grid(cards):
            grid = np.zeros(
                (len(Group18Player.suits), len(Group18Player.ranks)))
            for card in cards:
                np.put(grid[get_suit_index(card)], [get_rank_index(card)], 1)
            return grid

        def convert_to_image(starting_stack, round_state, street):
            image = np.zeros((2, 6))
            actions = round_state["action_histories"][street]
            index = 0
            turns = 0
            for action in actions:
                # max of 12 actions per street
                if 'amount' in action and turns < 6:
                    image[index, turns] = action['amount'] / starting_stack
                    index += 1

                if index % 2 == 0:
                    index = 0
                    turns += 1
            return image

        def get_actions_and_images():
            # initialization of all the images
            small_blind_cards_img = get_street_grid(hole_card)
            flop_cards_img = np.zeros((len(Group18Player.suits), len(Group18Player.ranks)))
            turn_cards_img = np.zeros((len(Group18Player.suits), len(Group18Player.ranks)))
            river_cards_img = np.zeros((len(Group18Player.suits), len(Group18Player.ranks)))

            # 2 x 6 matrix representation of all the actions
            flop_actions = np.zeros((2, 6))
            turn_actions = np.zeros((2, 6))
            river_actions = np.ones((2, 6))
            preflop_actions = convert_to_image(self.initial_stack, round_state,
                                               'preflop')

            if round_state['street'] == 'flop':
                flop_cards = round_state['community_card']
                flop_cards_img = get_street_grid(flop_cards)
                flop_actions = convert_to_image(self.initial_stack, round_state, 'flop')

            elif round_state['street'] == 'turn':
                turn_cards = [round_state['community_card'][3]]
                turn_cards_img = get_street_grid(turn_cards)
                turn_actions = convert_to_image(self.initial_stack, round_state, 'turn')

            elif round_state['street'] == 'river':
                river_cards = [round_state['community_card'][4]]
                river_cards_img = get_street_grid(river_cards)
                river_actions = convert_to_image(self.initial_stack, round_state, 'river')

            return {'actions': [preflop_actions, flop_actions, turn_actions, river_actions],
                    'images': [small_blind_cards_img, flop_cards_img, turn_cards_img, river_cards_img]}

        def run_q_learning_algorithm(y, max_replay_size, target_Q, old_small_blind_features):
            self.all_Q_small_blind = self.model.predict(self.small_blind_features)
            small_blind_action = np.argmax(self.all_Q_small_blind)
            small_blind_reward = 0

            if self.has_played:
                small_blind_reward += y * np.max(self.all_Q_small_blind)

                target_Q[0, small_blind_action] = small_blind_reward
                self.vvh = self.vvh + 1
                self.experience_state.append(old_small_blind_features)
                self.experience_reward.append(target_Q)
                if len(self.experience_state) > max_replay_size:
                    del self.experience_state[0]
                    del self.experience_reward[0]
            return small_blind_action

        def choose_small_blind_action(small_blind_action):
            learning_parameter = 0.1

            if np.random.rand(1) < learning_parameter:
                small_blind_action = np.random.randint(0, 4)

            if small_blind_action == 3 or len(valid_actions) == 2:
                small_blind_action = 1

            return small_blind_action

        def train_model():
            print(self.experience_state)
            for ve in range(len(self.experience_state)):
                self.model.fit(self.experience_state[ve],
                               self.experience_reward[ve],
                               verbose=0)

        # if the agent has already played an action before then its' previous attributes are stored for reference
        if self.has_played:
            Group18Player.old_small_blind_features = self.small_blind_features
            Group18Player.target_Q = self.all_Q_small_blind

        self.has_played = True

        actions_and_images = get_actions_and_images()

        actions = actions_and_images['actions']
        actions_feature = np.stack(actions, axis=2).reshape((1, 2, 6, 4))

        images = actions_and_images['images']
        sb_cards_feature = np.stack(images, axis=2).reshape((1, 4, 13, 4))

        self.small_blind_features = [
            sb_cards_feature, actions_feature,
            np.array([self.small_blind_position]).reshape((1, 1))
        ]

        # setting the hyper parameters
        y = 0.9
        max_replay_size = 40

        # run model to choose action
        small_blind_action = run_q_learning_algorithm(y, max_replay_size, Group18Player.target_Q,
                                                      Group18Player.old_small_blind_features)

        train_model()

        small_blind_action = choose_small_blind_action(small_blind_action)

        return valid_actions["action"][small_blind_action]

    def receive_game_start_message(self, game_info):
        self.initial_stack = game_info['rule']['initial_stack']

        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        self.small_blind_position = round_state['small_blind_pos']
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return Group18Player()
