from pypokerengine.players import BasePokerPlayer
import random as rand
import pprint
from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.emulator import Emulator
from pypokerengine.utils.game_state_utils import restore_game_state
import numpy as np
from keras.layers import Input, Dense, Conv2D,concatenate,Flatten
from keras.models import Model
import pandas as pd

class RandomPlayer(BasePokerPlayer):

    def __init__(self):
        self.table = {}
        self.my_uuid = None
        self.e = 0.1
        self.experience_state = []
        self.experience_reward = []
        self.play = False




    def declare_action(self, valid_actions, hole_card, round_state):
        def keras_model():

            input_cards = Input(shape=(4,13,4), name="cards_input")
            input_actions = Input(shape=(2,6,4), name="actions_input")
            input_position = Input(shape=(1,),name="position_input")

            x1 = Conv2D(32,(2,2),activation='relu')(input_cards)
            x2 = Conv2D(32,(2,2),activation='relu')(input_actions)
            x3 = Dense(1,activation='relu')(input_position)

            d1 = Dense(128,activation='relu')(x1)
            d1 = Flatten()(d1)
            d2 = Dense(128,activation='relu')(x2)
            d2 = Flatten()(d2)
            x = concatenate([d1,d2,x3])
            x = Dense(128)(x)
            x = Dense(32)(x)
            out = Dense(3)(x)

            model = Model(inputs=[input_cards, input_actions,input_position], outputs=out)
            model.compile(optimizer='rmsprop', loss='mse')

            #model.summary()

            return model

        def get_card_X(card):
            suit = card[0]
            if(suit == 'S'):
                return 0
            elif(suit == 'H'):
                return 1
            elif(suit=='D'):
                return 2
            elif(suit=='C'):
                return 3

        def get_card_Y(card):
            index = card[1]
            if(index=='A'):
                return 12
            elif(index=='K'):
                return 11
            elif(index=='Q'):
                return 10
            elif(index=='J'):
                return 9
            elif(index=='T'):
                return 8
            else:
                return int(index)-2

        def get_street_grid(cards):
            grid = np.zeros((4,13))
            for card in cards:
                grid[get_card_X(card),get_card_Y(card)] = 1
            return grid

        def convert_street_to_image(eff_stack,round_state,street):
            image = np.zeros((2,6))
            actions = round_state["action_histories"][street]
            index = 0
            turns = 0
            for action in actions:
            #max of 12actions per street
                if ('amount' in action and turns < 6):
                    image[index,turns] = action['amount'] / eff_stack
                    index += 1

                if(index%2 == 0):
                    index=0
                    turns +=1

            return image


        #bb_cards
        sb_cards = [hole_card[0], hole_card[1]]

        #bb_cards_img = get_street_grid(bb_cards)
        sb_cards_img = get_street_grid(sb_cards)
        flop_cards_img = np.zeros((4,13))
        turn_cards_img = np.zeros((4,13))
        river_cards_img = np.zeros((4,13))
        flop  = []
        turn  = []
        river = []

        #if round_state['small_blind_pos']['uuid']
        #self.my_uuid = round_state['next_player']['uuid']
        # print("my uuid")
        # for attr in dir(round_state['next_player']['uuid']):
        #     if hasattr( obj, attr ):
        #         print( "obj.%s = %s" % (attr, getattr(obj, attr)))
        self.my_uuid =  round_state['seats'][round_state['next_player']]['uuid']
        self.my_cards =  hole_card
        self.community_card = round_state['community_card']

        starting_stack = 10000

        if (self.play == True):
            self.old_state = self.sb_features
            self.targetQ = self.allQ_sb
            self.oldAction = self.action_sb

        sb_position = 1
        model = keras_model()
        flop_actions = np.zeros((2,6))
        turn_actions = np.zeros((2,6))
        river_actions = np.ones((2,6))

        preflop_actions = convert_street_to_image(starting_stack,round_state,'preflop')

        if (round_state['street'] == 'flop'):
            flop = round_state['community_card']
            flop_cards_img = get_street_grid(flop)
            flop_actions = convert_street_to_image(starting_stack,round_state,'flop')

        if ( round_state['street'] == 'turn'):
            turn = round_state['community_card'][2]
            turn_cards_img = get_street_grid([turn])
            turn_actions = convert_street_to_image(starting_stack,round_state,'turn')
        if (round_state['street'] == 'river'):
            river = round_state['community_card'][3]
            river_cards_img = get_street_grid([river])
            river_actions = convert_street_to_image(starting_stack,round_state,'river')

        self.actions_feature = np.stack([preflop_actions,flop_actions,turn_actions,river_actions],axis=2).reshape((1,2,6,4))
        sb_cards_feature = np.stack([sb_cards_img,flop_cards_img,turn_cards_img,river_cards_img],
                                    axis=2).reshape((1,4,13,4))
        # print("action_feature")
        # print(actions_feature.shape)
        # print("sb_cards_feature")
        # print(sb_cards_feature.shape)
        self.sb_features = [sb_cards_feature,self.actions_feature,np.array([sb_position]).reshape((1,1))]
        # print("combine")
        # print(self.sb_features.shape)
        #if round_state['seats'][round_state['big_blind_pos']]['uuid'] == self.my_uuid:
        y = 0.9
        max_replay_size = 50
        self.action_sb = 3
        #run model to choose action
        self.allQ_sb = model.predict(self.sb_features)
        self.action_sb = np.argmax(self.allQ_sb)
        reward_sb = 0
        if(self.play == True):
            reward_sb += y*np.max(self.allQ_sb)

            print("the reward is ")
            print(reward_sb)
            print(self.targetQ)
            print("this is old state")
            print(type(self.old_state))
            # dfObj = pd.DataFrame(self.old_state[0])
            # print(dfObj.shape)
            self.targetQ[0,self.action_sb] = reward_sb
            #model.fit(self.old_state,self.targetQ,verbose=0)
            self.experience_state.append(self.old_state)
            self.experience_reward.append(self.targetQ)
            if(len(self.experience_state) > max_replay_size):
                del self.experience_state[0]
                del self.experience_reward[0]

        self.play = True

        e = 0.1

        for ve in range(len(self.experience_state)):
            print("hello")
            model.fit(self.experience_state[ve],self.experience_reward[ve],verbose = 0)
            

        if(np.random.rand(1) < e):
            self.action_sb = np.random.randint(0,4)

        if (self.action_sb == 3 or len(valid_actions ) == 2):
            self.action_sb = 1

        if(self.action_sb == 0):
            #game_state,events = emulator.apply_action(game_state,'fold',0)
            call_action_info = valid_actions[0]
            action = call_action_info["action"]
            return action

        if(self.action_sb == 1):
            call_action_info = valid_actions[1]
            action = call_action_info["action"]
            return action


        if(self.action_sb == 2):
            call_action_info = valid_actions[2]
            action = call_action_info["action"]
            return action



        # print(round_state['small_blind_pos'])
        #
        # for g in round_state['action_histories']:
        #     print(g)
        # #if round_state['small_blind_pos'] == 0:





        r = rand.random()
        if r <= 0.5:
            call_action_info = valid_actions[1]
        elif r<= 0.9 and len(valid_actions ) == 3:
            call_action_info = valid_actions[2]
        else:
            call_action_info = valid_actions[0]
        action = call_action_info["action"]
        return action  # action returned here is sent to the poker engine






    def receive_game_start_message(self, game_info):

        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        # print("winners")
        # for i in winners:
        #
        #     print(i)

        pass

def setup_ai():
  return RandomPlayer()
