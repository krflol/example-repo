from fastapi import APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import multiprocessing
import operator
from pydantic import BaseModel
import torch
from collections import deque
import ast
import time
import requests
import json
import os
import datetime
import time
import pandas as pd
import numpy as np
import random
#from somcomes import RLSOM
from ppocalvin import device, PPOAgent as PPOCalvin
import datetime as dt
from routers.calvin.bnn_trader import bnn
from .datarecorder import write_state_and_spread_to_csv
from candlestick import candlestick
from collections import deque
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import defaultdict


#from venv.share.doc.networkx-3.1.examples.algorithms.plot_beam_search import value
router = APIRouter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Specify the device for computation
#calvin params
#calvin = RLSOM()
agent = PPOCalvin(state_dim=70, action_dim=6, model_dir="path_to_save_model").to(device)
agent.save_models()
agent.to(agent.device)
#closing_agent = PPOCalvin(state_dim=5, action_dim=2, model_dir="closing_model")
#calvin.eps = 0
threshold_start = 100
threshold_end = 9999

#a dictioary of states to determine if there is currently a trade active at the time of the request
#this is used to ensure multiple trades are not active at the same time at the same state

is_state_active = {}
action_probs_old = None
log_probs_old = None
pattern_exists = False
buy_signal = False
sell_signal = False
sliding_state = []
is_buyer = True
is_seller = True
buyer_actions = [0,5,4]
seller_actions = [1,2,3]

def som_output_to_state_index(som_coordinates):
        x, y = som_coordinates
        grid_width = 10  # Assuming a 10x10 SOM grid
        return y * grid_width + x

class Observation:
    state:list
    action:int
    reward:float
    next_state:list
    log_probs_old:float
    value_old = None
    done:bool

class Memory:
    def __init__(self):
        self.states = deque(maxlen=10)
        self.actions = deque(maxlen=10)
        self.rewards = deque(maxlen=10)
        self.next_states = deque(maxlen=10)
        self.log_probs_old= deque(maxlen=10)
        self.values_old = deque(maxlen=10)
        self.dones = deque(maxlen=10)

    def add_experience(self, state, action, reward, next_state, done, log_probs_old, value_old):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs_old.append(log_probs_old)
        self.values_old.append(value_old)
    
    def to_csv(self):
        # Convert the deque of tensors to a list of NumPy arrays
        #values_old_list = [tensor.detach().cpu().numpy().flatten() for tensor in self.values_old]

        # Save the list of arrays to a CSV file
            # Save the list of arrays to a CSV file in append mode
        with open("memory.csv", "a") as file:
        #    np.savetxt(file, values_old_list, delimiter=",")
            np.savetxt(file, self.states[-1], delimiter=",")

    def clear_memory(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.values_old.clear()
        self.log_probs_old.clear()

memory = Memory()

observation = Observation()

@router.post("/get_action_endpoint")

def get_action_endpoint(buys_1: float,
                        sells_1: float, 
                        spread_1: float, 
                        time_to_close_1: float, 
                        buys_2: float,
                        sells_2: float,
                        spread_2: float,
                        time_to_close_2: float,
                        buys_3: float,
                        sells_3: float,
                        spread_3: float,
                        time_to_close_3: float,
                        poc_dist: float, 
                        vah_dist: float,
                        val_dist: float,                      
                        value_threshold: float,
                        eps: float, 
                        confidence_threshold:  float,
                        is_flat: bool, 
                        open_pnl: float,
                        ):
    try:
        agent.load_models()

    except Exception as e:
        print(e)
        pass
    #print(time_to_close)
    #winner = calvin.get_winner(buys, sells, spread, time_to_close)
    #print(winner)
    #get random number between 1 and 100
    random_number = random.randint(1,100)
    #check if the current time is between 3pm and 5pm
    the_time = dt.datetime.now()
    winner = [buys_2, sells_2, spread_2, time_to_close_2,poc_dist]
    winner = agent.scaler.transform([winner])[0]
    #winner = agent.som.winner(winner)
    #winner = som_output_to_state_index(winner)

    #winner = list(agent.actor.scaler.transform([winner])[0])
    #winner = agent.actor.som.winner(winner)
    #can_trade_time = True #if the_time.hour <= 15 and the_time.hour >= 17 else False
    #if winner not in is_state_active: #TODO add a number of acceptable trades per state
    #    is_state_active[winner] = False
    #if random number is greater than 98, get the action from the agent
    #action = calvin.get_action(90,110)#TODO threshold
    
    #scale open_pnl to be between -1 and 1. min input is -600, max input is 1000
    open_pnl = (open_pnl + 600) / 1600
    print(winner)
    global sliding_state
    winner = sliding_state

    #if is_state_active[winner] == False and can_trade_time:
    if len(sliding_state) == 70:
        
        action, winner, observation.log_probs_old,observation.value_old, action_probs = agent.act(winner,False)#TODO threshold
        winner = winner.tolist()
        observation.state = winner
        observation.action = action
        #value = observation.value_old[0].item()
        #value = agent.value_scalar.transform(np.array([value]).reshape(1, -1))[0]
        #value = agent.value_som.winner(value)[0]
        

        if observation.value_old[0] < value_threshold:
            action = 6
        if is_buyer == False and action in buyer_actions:
            action = 6
        if is_seller == False and action in seller_actions:
            action = 6
        
        print(action)
        #if action != 2:
        #    is_state_active[winner] = True
        #    #print(is_state_active)
        #else:
        #    action = 2
        #    value = 0
        #print({"action": action, "winner": winner})
        #return winner and action as a json response
        #global pattern_exists
        #if pattern_exists == False:
        #    action = 2
        #if action == 0 and buy_signal == True:
        #    action = 0
        #elif action == 1 and sell_signal == True:
        #    action = 1
        #if action == 0 and sell_signal == True:
        #    action = 2
        #elif action == 1 and buy_signal == True:
        #    action = 2
        #
        print(observation.value_old[0])
        #with open("memory.csv", "a") as file:
        #    np.savetxt(file, [value], delimiter=",")
        #print(value)
        return {"action": action, "winner": f"{winner}", "value": f"0"}
    else:
        return {"action": 6, "winner": f"{winner}", "value": 0}

experiences = []
memory_per_agent = defaultdict(Memory)


BATCH_SIZE = 6  # The number of experiences to sample from the replay buffer

@router.post("/update_agent_endpoint")
def update_agent_endpoint(reward: float,
                            action: int,
                            state: str,
                            done: bool,
                            agent_id:str,
                            buys,
                            sells,
                            spread,
                            time,
                            poc_dist):
    # Convert state from string to list using ast.literal_eval
    #state = [buys, sells, spread, time, poc_dist]
    #state = agent.scaler.transform([state])[0]
    state_list = ast.literal_eval(state)
    #state = agent.som.winner(state)
    #state = som_output_to_state_index(state)
    #state = torch.nn.functional.one_hot(torch.tensor([state]), num_classes=100).float().to(device)
    #state_list = state.tolist()

   #if done == False:
   #    #state_list = ast.literal_eval(state)
   #    hold_or_close, _,   _,_,_ = agent.act(sliding_state,True)
   #     
   #    #write_state_and_spread_to_csv(state_list, spread)
   #else:
   #    hold_or_close, _,   _,_,_ = agent.act(sliding_state,False)
    # Add current experience to memory
    #if hasattr(observation, 'next_state'):

    # Replace with the new state and other details
    #observation.state = state_list
    observation.action = action
    observation.reward = reward
    observation.next_state = state_list  # Not sure why this is the same as state_list, needs clarification

    observation.done = done
    memory.add_experience(observation.state,
                           observation.action,
                            observation.reward,
                            observation.next_state,
                            observation.done,
                            observation.log_probs_old,
                            observation.value_old)

    

    # Check if we have enough experiences in memory to perform an update
    if len(memory.states) >= BATCH_SIZE:
        agent.update(
            observations=memory.states,
            actions=memory.actions,
            rewards=memory.rewards,
            next_observations=memory.next_states,
            dones=memory.dones,
            log_probs_old=memory.log_probs_old,
            values_old=memory.values_old
            
        )
        #memory.to_csv()

        memory.clear_memory()  # Clear memory after update
        agent.save_models()


    return {"action": 6, "winner": f"{state}", "value": f"0"}

def reverse_action(action):
    if action == 0:
        return 1
    elif action == 1:
        return 0

#def update_agent_endpoint(reward: float, action: int, winner: str, current_state: str):
#    
#    global last_state, experiences  # Ensure we're modifying the global variables
#    #calvin.update_agent(reward=reward, action=action, winner=ast.literal_eval(winner))
#    # Convert the winner state to a tensor (this operation is small and necessary for the next step)
#    winner_state = torch.tensor(ast.literal_eval(winner), dtype=torch.float64).to(device)
#    experiences.append((winner_state, action, reward, current_state))  # store as strings or native types
#
#    # If this isn't the first state and we have a valid last state, store the experience
#        #normalize the reward max is 1000 min is -1000
#    #reward = (reward + 5000) / 10000
#        
#        # Append the new experience. If experiences is already at BATCH_SIZE, 
#        # the oldest experience will automatically be removed.
#        #make reward reverse. if action is 0, make action 1. if action is 1 make action 0 and append to experiences
#    if action == 0:
#       experiences.append((winner_state, 1, -reward, current_state))
#    elif action == 1:
#       experiences.append((winner_state, 0, -reward, current_state))
#        
#
#    # Update the last state with the current winner state (state when the action was taken)
#    last_state = winner_state
#
#    # If we have enough experiences to form a batch, we update the agent.
#    if len(experiences) == BATCH_SIZE:
#        #agent.load_model()
#        # Now we need the experiences as tensors, so we convert them here.
#        states, actions, rewards, next_states = [], [], [], []
#        for exp in experiences:
#            states.append(exp[0])  # exp[0] is already a tensor, so we just append it
#            actions.append(exp[1])  # Assuming actions are integers or can be directly converted to tensor
#            rewards.append(exp[2])  # Assuming rewards are floats or can be directly converted to tensor
#            #next_states.append(torch.tensor(ast.literal_eval(exp[3]), dtype=torch.float32).to(device))
#
#        # Convert lists to tensors
#        states = torch.stack(states)
#        actions = torch.tensor(actions, dtype=torch.long, device=device)
#        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
#        #next_states = torch.stack(next_states)
#
#        # Calculate the 'done' flags here. This depends on your environment.
#        # For simplicity, we're assuming they're all False, meaning the episode hasn't ended.
#        #dones = torch.tensor([False] * len(rewards), dtype=torch.bool, device=device)
#
#        # Now we can update the agent with a batch of experiences
#        #agent.update_agent(states, actions, rewards, next_states, dones)
#        agent.update_agent(states, actions, rewards)
#
#        #closing_agent.update_agent(states, actions, rewards, next_states, dones)
#        experiences.clear()  # clear the experiences list after processing
# 
#
#    return {"status": "success"}
#TODO pattern recognition
queue_length = 10
opens = deque(maxlen=queue_length)
highs = deque(maxlen=queue_length)
lows = deque(maxlen=queue_length)
closes = deque(maxlen=queue_length)
volumes = deque(maxlen=queue_length)
buy_volumes = deque(maxlen=queue_length)
sell_volumes = deque(maxlen=queue_length)
spreads = deque(maxlen=queue_length)
time_to_closes = deque(maxlen=queue_length)
market_states = deque(maxlen=queue_length)
poc_dists = deque(maxlen=queue_length)
vah_dists = deque(maxlen=queue_length)
val_dists = deque(maxlen=queue_length)
@router.post("/get_candlestick")
def get_candlestick(open_price: float,
                    high: float,
                    low: float,
                    close: float,
                    volume:float,
                    buy_volume:float,
                    sell_volume:float,
                    spread:float,
                    time_to_close:float,
                    poc_dist:float,
                    vah_dist:float,
                    val_dist:float,
                    trade_open: bool):

    opens.append(open_price)
    highs.append(high)
    lows.append(low)
    closes.append(close)
    volumes.append(volume)
    buy_volumes.append(buy_volume)
    sell_volumes.append(sell_volume)
    spreads.append(spread)
    time_to_closes.append(time_to_close)
    poc_dists.append(poc_dist)
    vah_dists.append(vah_dist)
    val_dists.append(val_dist)

    #market_states.append(agent.scaler.transform([buy_volume, sell_volume, spread, time_to_close])[0].tolist())
    global pattern_exists
    global buy_signal
    global sell_signal
    pattern_exists = False
    buy_signal = False
    sell_signal = False
    global sliding_state
    action = 6
    if len(opens) == queue_length:
        df = pd.DataFrame()
        #df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes})
        ##devide market_state and make a column for each
        #df = candlestick.rising_three_methods(df,target="rising_three_methods")
        #df = candlestick.falling_three_methods(df,target="falling_three_methods")
        #df = candlestick.doji(df,target="doji")
        #df = candlestick.shooting_star(df,target="shooting_star")
        #df = candlestick.morning_star(df,target="morning_star")
        #df = candlestick.three_white_soldiers(df,target="three_white_soldiers")
        scaler = StandardScaler()
        opens_normalized = scaler.fit_transform(np.array(opens).reshape(-1, 1)).flatten()
        #df["open"] = opens_normalized
        #highs_normalized = scaler.fit_transform(np.array(highs).reshape(-1, 1)).flatten()
        #df["high"] = highs_normalized
        #lows_normalized = scaler.fit_transform(np.array(lows).reshape(-1, 1)).flatten()
        #df["low"] = lows_normalized
        #closes_normalized = scaler.fit_transform(np.array(closes).reshape(-1, 1)).flatten()
        #df["close"] = closes_normalized
        #volumes_normalized = scaler.fit_transform(np.array(volumes).reshape(-1, 1)).flatten()
        #df["volume"] = volumes_normalized
        buy_volumes_normalized = scaler.fit_transform(np.array(buy_volumes).reshape(-1, 1)).flatten()
        df["buy_volume"] = buy_volumes_normalized
        sell_volumes_normalized = scaler.fit_transform(np.array(sell_volumes).reshape(-1, 1)).flatten()
        df["sell_volume"] = sell_volumes_normalized
        spreads_normalized = scaler.fit_transform(np.array(spreads).reshape(-1, 1)).flatten()
        df["spread"] = spreads_normalized
        time_to_closes_normalized = scaler.fit_transform(np.array(time_to_closes).reshape(-1, 1)).flatten()
        df["time_to_close"] = time_to_closes_normalized
        poc_dists_normalized = scaler.fit_transform(np.array(poc_dists).reshape(-1, 1)).flatten()
        df["poc_dist"] = poc_dists_normalized
        vah_dists_normalized = scaler.fit_transform(np.array(vah_dists).reshape(-1, 1)).flatten()
        df["vah_dist"] = vah_dists_normalized
        val_dists_normalized = scaler.fit_transform(np.array(val_dists).reshape(-1, 1)).flatten()
        df["val_dist"] = val_dists_normalized



        #convert true or false to 1 or 0
        df = df.replace(True, 1)
        df = df.replace(False, 0)
        #replace nan with 0
        df = df.fillna(0)
        sliding_state = df.values.tolist()
        #flatten the list
        #candlestick_state = [opens_normalized, highs_normalized, lows_normalized, closes_normalized, volumes_normalized, buy_volumes_normalized, sell_volumes_normalized, spreads_normalized, time_to_closes_normalized, poc_dists_normalized, vah_dists_normalized, val_dists_normalized]
        candlestick_state = [buy_volumes_normalized, sell_volumes_normalized, spreads_normalized, time_to_closes_normalized, poc_dists_normalized, vah_dists_normalized, val_dists_normalized]
        #candlestick_state = [buy_volume, sell_volume, spread, time_to_close, poc_dist, vah_dist, val_dist]
        #write to memory.csv
        #create a header if the file doesnt exist
        #if not os.path.isfile("memory.csv"):
        #    with open("memory.csv", "a") as file:
        #        file.write("buy_volume,sell_volume,spread,time_to_close,poc_dist,vah_dist,val_dist\n")
        ##write the state to memory.csv
        #with open("memory.csv", "a") as file:
        #    file.write(f"{buy_volume},{sell_volume},{spread},{time_to_close},{poc_dist},{vah_dist},{val_dist}\n")
        #sliding_state = [item for sublist in sliding_state for item in sublist]
        sliding_state = [item for sublist in candlestick_state for item in sublist]
        print(len(sliding_state))
        #sliding_state = np.concatenate([opens_normalized, highs_normalized, lows_normalized, closes_normalized, volumes_normalized]).tolist()
        #if the result column contains a True, then return True

        #plot the dataframe in real time using matplotlib
               
        #if True in df["rising_three_methods"].values or True in df["morning_star"].values or True in df["three_white_soldiers"].values:
        #    pattern_exists= True
        #    buy_signal = True
        #    
        #    return {"result": True}
        #if True in df["falling_three_methods"].values or True in df["shooting_star"].values:
        #    pattern_exists= True
        #    sell_signal = True
        #    return {"result2": True}
        #if True in df["doji"].values:
        #    pattern_exists= True#TODO change to true
        #    return {"result3": True}
        #else:
        #    pattern_exists = False
        #    return {"result": False}
        
        if trade_open == True:
            action, _,   _,_,_ = agent.act(sliding_state,True)
            return {"action": action,"winner": f"{sliding_state}", "value":0 }
        else:
            action = 6
    return {"action": action,"winner": f"{sliding_state}", "value":0 }

@router.post("/buyer_seller")
def set_buyer_seller(buyer: bool, seller: bool):
    global is_buyer
    global is_seller
    is_buyer = buyer
    is_seller = seller
    return {"buyer": is_buyer, "seller": is_seller}






