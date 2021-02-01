# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 16:20:50 2020

@author: alire
"""
import os
import sys

import numpy as np
from CreateData import dataFile

#from gym import core, spaces
import core
from box import Box
from discrete import Discrete

#from gym.utils import seeding
import seeding



class PMSP1:
    
    
    def __init__(self, EVs=4, horizon=24, slot=1):
        self.N=EVs
        self.T=horizon*slot
        self.timer=0
        self.slot=slot
        self.act_dict=dict()
                
        
        self.state=None
        self.seed()
        
        #Create Scenario
        self.scenario()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):

        self.state = np.ones([self.M, self.T])*-1
        
        #handle scheduled EVs
        self.is_schedule_ev=np.zeros(self.N)
        
        #Handle installed chargers
        self.is_installed_ch = np.zeros(self.M)
        
        self.timer=0
        
    
        return self._get_ob()
    
    def _get_ob(self):
        #s = self.state.tolist()
        s = self.state
        return s
    
    def _terminal(self): 
        return self.all_evs_charged()
    
    def step(self, a):
        s = self.state
        
        # #update state and installed EVs and Chargers
        # for act in a:
        #     ev=self.actions_list[act][0]
        #     ch=self.actions_list[act][1]
        #     if ev!=-1:
        #         self.is_schedule_ev[ev]=1
        #         self.is_installed_ch=1
        #         time=self.timer
        #         for i in range(time, time+self.TFC[ev][ch]+1):
        #             s[ev][i]=ev
        
        #update state and installed EVs and Chargers
       
        ev=self.actions_list[a][0]
        ch=self.actions_list[a][1]
        if ev!=-1:
            self.is_schedule_ev[ev]=1
            self.is_installed_ch[ch]=1
            time=self.timer
            for i in range(time, int(time+self.TFC[ev][ch])):
                print(ev," ",ch," ",i)
                s[ch][i]=ev
        
        
        reward=self.reward(a)
        self.state=s
        terminal=self._terminal()
                       
        #update timer after finishing each step
        self.timer+=1
        
        if self.timer >= self.T:
            self.timer=0
        
        
        return (self._get_ob(), reward, terminal, {})
    
    # def valid_actions(self):
    #     time=self.timer
    #     s=self.state
    #     chargers=[]
    #     evs=[]
    #     actions=[]
        
    #     #Listing Availabe Chargers
    #     for j in range(self.M):
    #         if s[j][time]==-1:
    #             chargers.append(j)
        
    #     for i in range(self.N):
    #         if self.is_schedule_ev[i]==0 and self.arrival[i]<=time:
    #             evs.append(i)
        
    #     temp=[]
    #     for i in evs:
    #         temp=[]
    #         for j in chargers:
    #             if self.TFC[i][j]+time <= self.T:
    #                 if all(x==True for x in (self.state[j][time:self.TFC[i][j]+time]==-1)):
    #                     temp.append(self.act_dict[(i,j)])
    #         actions.append(temp)
        
    #     temp=[]
    #     for j in chargers:
    #         temp.append(self.act_dict[(-1,j)])
        
    #     actions.append(temp)
        
    #     actions=np.array(actions).T
        
    #     return actions

    def valid_actions(self):
        time=self.timer
        s=self.state
        chargers=[]
        evs=[]
        actions=[]
        
        #Listing Availabe Chargers
        for j in range(self.M):
            if s[j][time]==-1:
                chargers.append(j)
        
        for i in range(self.N):
            if self.is_schedule_ev[i]==0 and self.arrival[i]<=time:
                evs.append(i)
        
        
        for i in evs:
            temp=[]
            for j in chargers:
                if self.TFC[i][j]+time < self.T:
                    sl=int(self.TFC[i][j]+time)
                    if all(x==True for x in (self.state[j][time:sl]==-1)):
                        actions.append(self.act_dict[(i,j)])
            
        
        
        for j in chargers:
            actions.append(self.act_dict[(i,j)])
        
        
        return actions        
        
        
    def scenario(self):
        
        slot=self.slot   #Add more time slot for more accurate result

        number_of_EVs=self.N

        number_of_Chargers=0  #took it from dataFile output later
           
        number_of_timeslot=24*self.slot

        #Charger_Type=[4, 8, 19, 50]     #type of chargers to install
        Charger_Type=[4, 8, 19]

        #charger_cost=[1000,1500,2200, 50000]  #cost of installation
        charger_cost=[1000,1500,2200] 
        
        self.arrival, self.depart, self.distance, self.demand, self.charge_power,self.installed_chargers,\
             self.installed_cost,self.TFC, self.EV_samples = dataFile(number_of_EVs,
                                                       number_of_timeslot,
                                                       Charger_Type,
                                                       charger_cost,
                                                       slot)
             
         
        #number of required chargers
        self.M = len(self.installed_chargers)
        
        low  = -2 * np.ones([self.M, self.T])
        high =  self.N * np.ones([self.M, self.T])
        
        self.observation_space = Box(low=low, high=high,shape=None, dtype=np.uint8)
        self.action_space = Discrete(self.N*self.M + self.M) # Must chage to new Class
        
        #Create dictionary for avilable actions to perform
        self.actions_dict()
        
        pass
        
        
    def actions_dict(self):
         count=0
         
         for i in range(self.N):
             for j in range(self.M):
                 self.act_dict[(i,j)]=count
                 count+=1
        
         for j in range(self.M):
            self.act_dict[(-1,j)]=count
            count+=1
        
         self.actions_list = list(self.act_dict.keys())
         print(self.act_dict)
         pass
                 
    def all_evs_charged(self):
        
        result=np.where(self.is_schedule_ev==0)
        if result:
            return False
        else:
            return True
        pass
        
    def reward(self, a):
        reward=0
         
        # #If it's terminal position find the final reward
        # if self._terminal():
        #     reward=sum(x*y for x , y in zip(self.is_installed_ch, self.installed_cost))
        #     #else:
        #      #   reward= -sys.maxsize-1
        # else:
        #     for act in a:
        #         ev=self.actions_list[act][0]
        #         ch=self.actions_list[act][1]
        #         if ev!=-1:
        #             reward += 10*min(0,self.timer+self.TFC[ev][ch]-self.depart[ev])
        
        #If it's terminal position find the final reward
        if self._terminal():
            reward=sum(x*y for x , y in zip(self.is_installed_ch, self.installed_cost))
            #else:
             #   reward= -sys.maxsize-1
        else:
            ev=self.actions_list[a][0]
            ch=self.actions_list[a][1]
            if ev!=-1:
                reward += 10*min(0,self.timer+self.TFC[ev][ch]-self.depart[ev])
                    
        return reward
         
    
    
"""
New Class
"""


        
        
        