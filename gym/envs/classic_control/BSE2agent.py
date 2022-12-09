from cgitb import reset
from random import randint
from re import T
from exchange import *
import numpy as np

##########################---Below lies the experiment/test-rig---##################
from datetime import datetime
current_time = datetime.now().strftime("%m%d_%H-%M-%S")


class Agent2BSE:

    def __init__(self,training=True,range2_input=None,traders_per=None,**kargs):
        
        # flag
        self.orders_verbose = False
        self.lob_verbose = False
        self.process_verbose = False
        self.respond_verbose = False
        # bookkeep_verbose = True
        self.bookkeep_verbose = False
        self.populate_verbose = False

        self.traders_per = traders_per
        
        self.training = training

        self.start_time = 0.0
        self.num=-10000

        # TODO 总时间
        if self.training:
            self.end_time = 100.0
            self.num = 0
        else:
            self.end_time = 3000.0
            self.num=-1000000

        #TODO end_time
        # self.end_time = 360000

        self.duration = self.end_time - self.start_time

        self.time_record = 0

        self.agent_tid = -1

        self.minprice=0
        self.maxprice=500


        self.len_lobs = 20
        self.windows = 20
        self.lobs_windows = [[[self.maxprice]*self.len_lobs]*self.windows,
                                [[self.minprice]*self.len_lobs]*self.windows]
        #length,batchsize,vit
        # self.lobs_windows_lstm = [[[self.maxprice]*self.len_lobs+
        #                         [self.minprice]*self.len_lobs] for _ in range(self.windows)]
        self.lobs_windows_lstm = [[self.maxprice]*self.len_lobs+
                                [self.minprice]*self.len_lobs for _ in range(self.windows)]


        self.rew=0

        self.range1_min = 100
        self.range1_int = 50
        range1 = (self.range1_min,self.range1_min+self.range1_int) #卖
        # supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}
        #                    ]
        supply_schedule = [{'from': self.start_time, 'to': self.end_time, 'ranges': [range1], 'stepmode': 'random'}
                        ]

        # range2 = (350,400) #买

        # TODO ragne2
        self.range2_min = 350
        # self.range2_min = 250
        self.range2_min = np.random.randint(150,350)
        self.range2_int = 50

        self.range2_input = range2_input

        if self.range2_input:
            self.range2_min = range2_input

        range2 = (self.range2_min,self.range2_min+self.range2_int)
        # demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}
        #                    ]
        demand_schedule = [{'from': self.start_time, 'to': self.end_time, 'ranges': [range2], 'stepmode': 'random'}
                        ]

        self.range_milit = [self.range1_min,self.range1_int,self.range2_min,self.range2_int]

        self.order_schedule = {'sup': supply_schedule, 'dem': demand_schedule,
                    'interval': 30, 'timemode': 'drip-poisson'}

        a1=np.random.randint(30)
        b1=np.random.randint(30-a1)
        c1=30-a1-b1

        a2=np.random.randint(30)
        b2=np.random.randint(30-a2)
        c2=30-a2-b2

        # TODO trader随机
        # if self.training:
        buyers_spec = [('GVWY',a1),('ZIC',b1),('SHVR',c1)]
        sellers_spec = [('GVWY',a2),('ZIC',b2),('SHVR',c2),('Agent',1)]    
        # else:
        #     buyers_spec = [('GVWY',10),('ZIC',10),('SHVR',10)]
        #     sellers_spec = [('GVWY',10),('ZIC',10),('SHVR',10),('Agent',1)]

        if self.traders_per and not self.training:
            a1,b1,c1,a2,b2,c2 = self.traders_per
            buyers_spec = [('GVWY',a1),('ZIC',b1),('SHVR',c1)]
            sellers_spec = [('GVWY',a2),('ZIC',b2),('SHVR',c2),('Agent',1)] 
        # else:
        #     buyers_spec = [('GVWY',10),('ZIC',10),('SHVR',10)]
        #     sellers_spec = [('GVWY',10),('ZIC',10),('SHVR',10),('Agent',1)]

        buyers_spec = [('GVWY',10),('ZIC',10),('SHVR',10)]
        sellers_spec = [('GVWY',10),('ZIC',10),('SHVR',10),('Agent',1)]
        # sellers_spec = [('GVWY',10),('ZIC',10),('SHVR',10),('Agent',1),('PRSH',1),('AA',1),('ZIP',1)]

        traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}

        # run a sequence of trials, one session per trial

        verbose = True

        # n_trials is how many trials (i.e. market sessions) to run in total
        n_trials = 1

        # n_recorded is how many trials (i.e. market sessions) to write full data-files for
        n_trials_recorded = 3

        # tdump=open('avg_balance.csv','w')

        # initialise the exchange
        self.exchange = Exchange()

        # create a bunch of traders
        traders = {}

        trader_stats = populate_market(traders_spec, traders, True, self.populate_verbose, self.range_milit)

        self.traders = traders
        self.trader_stats = trader_stats

        # timestep set so that can process all traders in one second
        # NB minimum interarrival time of customer orders may be much less than this!!
        timestep = 1.0 / float(trader_stats['n_buyers'] + trader_stats['n_sellers'])

            
        self.last_update = -1.0

        self.time = self.start_time
        self.timestep = 1.0 / float(trader_stats['n_buyers'] + trader_stats['n_sellers'])
        self.time_left = (self.end_time - self.time) / self.duration

        self.pending_cust_orders = []


    def lob_update(self):

        dir_sort_asks = sorted(self.exchange.asks.lob.items(),key=lambda x:x[0])
        dir_sort_bids = sorted(self.exchange.bids.lob.items(),key=lambda x:x[0],reverse=True)
        lob_asks,lob_bids = [],[]
        num_asks,num_bids = 0,0
        index_asks,index_bids = 0,0
        while num_asks < self.len_lobs:
            if index_asks+1>len(dir_sort_asks):
                lob_asks.extend([self.maxprice]*(self.len_lobs-num_asks))
                num_asks += self.len_lobs-num_asks
            else:
                lob_asks.extend([dir_sort_asks[index_asks][0]]*min(self.len_lobs-num_asks,dir_sort_asks[index_asks][1][0]))
                num_asks += min(self.len_lobs-num_asks,dir_sort_asks[index_asks][1][0])
            index_asks+=1
        

        while num_bids < self.len_lobs:
            if index_bids+1>len(dir_sort_bids):
                lob_bids.extend([self.minprice]*(self.len_lobs-num_bids))
                num_bids += self.len_lobs-num_bids
            else:
                lob_bids.extend([dir_sort_bids[index_bids][0]]*min(self.len_lobs-num_bids,dir_sort_bids[index_bids][1][0]))
                num_bids += min(self.len_lobs-num_bids,dir_sort_bids[index_bids][1][0])
            index_bids+=1
        
        #TODO record price
        # bdump = open(f'data_temp_ask{current_time}.csv', 'a')
        # bdump.write(f'{self.time},{lob_asks[0]},{lob_bids[0]}\n')
        # bdump.close()


        if lob_asks != self.lobs_windows[0][-1] or lob_bids != self.lobs_windows[1][-1]:
            self.lobs_windows[0][:-1] = self.lobs_windows[0][1:]
            self.lobs_windows[0][-1] = lob_asks
            self.lobs_windows[1][:-1] = self.lobs_windows[1][1:]
            self.lobs_windows[1][-1] = lob_bids

        if lob_asks+lob_bids != self.lobs_windows_lstm[-1]:
            self.lobs_windows_lstm[:-1] = self.lobs_windows_lstm[1:]
            self.lobs_windows_lstm[-1] = lob_asks+lob_bids
        

    def _get_obs(self):
        dir_sort_asks = sorted(self.exchange.asks.lob.items(),key=lambda x:x[0])
        dir_sort_bids = sorted(self.exchange.bids.lob.items(),key=lambda x:x[0],reverse=True)
        lob_asks,lob_bids = [],[]
        num_asks,num_bids = 0,0
        index_asks,index_bids = 0,0
        while num_asks < self.len_lobs:
            if index_asks+1>len(dir_sort_asks):
                lob_asks.extend([self.maxprice]*(self.len_lobs-num_asks))
                num_asks += self.len_lobs-num_asks
            else:
                lob_asks.extend([dir_sort_asks[index_asks][0]]*min(self.len_lobs-num_asks,dir_sort_asks[index_asks][1][0]))
                num_asks += min(self.len_lobs-num_asks,dir_sort_asks[index_asks][1][0])
            index_asks+=1
        

        while num_bids < self.len_lobs:
            if index_bids+1>len(dir_sort_bids):
                lob_bids.extend([self.minprice]*(self.len_lobs-num_bids))
                num_bids += self.len_lobs-num_bids
            else:
                lob_bids.extend([dir_sort_bids[index_bids][0]]*min(self.len_lobs-num_bids,dir_sort_bids[index_bids][1][0]))
                num_bids += min(self.len_lobs-num_bids,dir_sort_bids[index_bids][1][0])
            index_bids+=1
        
        #TODO obs
        # return [lob_asks, lob_bids]
        # return self.lobs_windows
        return self.lobs_windows_lstm

    def _get_rew(self):
        reward = self.traders[self.agent_tid].profit_last
        self.traders[self.agent_tid].profit_last = 0
        # if self.traders[self.agent_tid].profit_last!=0:
        #     print(self.traders[self.agent_tid].profit_last)
        if reward==0 and self.training:
            reward = - 0.1
        # print(reward)
        return reward

    def agent_process_order(self, action):

        order = self.traders[self.agent_tid].getorder(self.time, self.time_left, self.exchange.publish_lob(self.time, self.lob_verbose), action)
        
        if order is not None:
            # print(order.price,self.traders[self.agent_tid].orders[0].price)
            # if order.otype == 'Ask' and order.price < self.traders[self.agent_tid].orders[0].price:
            #     sys.exit('Bad ask')
            # if order.otype == 'Bid' and order.price > self.traders[self.agent_tid].orders[0].price:
            #     sys.exit('Bad bid')
            # send order to exchange

            #TODO record agent price
            # if self.traders[tid].ttype in ['Agent','AA','PRSH','ZIP']:
            # bdump = open(f'data_temp_Agent{current_time}.csv', 'a')
            # bdump.write(f'{self.time},{order.price},\n')
            # bdump.close()


            self.traders[self.agent_tid].n_quotes = 1
            trade = self.exchange.process_order2(self.time, order, self.process_verbose)
            if trade is not None:
                # trade occurred,
                # so the counterparties update order lists and blotters
                self.traders[trade['party1']].bookkeep(trade, order, self.bookkeep_verbose, self.time)
                self.traders[trade['party2']].bookkeep(trade, order, self.bookkeep_verbose, self.time)
                
                # 记录
                # if dump_all:
                #     trade_stats(sess_id, self.traders, tdump, self.time, self.exchange.publish_lob(time, lob_verbose))

            # traders respond to whatever happened
            lob = self.exchange.publish_lob(self.time, self.lob_verbose)
            for t in self.traders:
                self.traders[t].respond(self.time, lob, trade, self.respond_verbose)
            self.lob_update()


        self.time = self.time + self.timestep

        # 继续进行交易
    def continues(self):

        verbose = True
        while self.time < self.end_time:

            self.time_left = (self.end_time - self.time) / self.duration # 有一些策略需要根据距离结束的时间进行权重更改
            trade = None
            [self.pending_cust_orders, kills, self.num, flag] = customer_orders(self.time, self.last_update, self.traders, self.trader_stats,self.order_schedule, self.pending_cust_orders, self.orders_verbose,num=self.num)

            # if any newly-issued customer orders mean quotes on the LOB need to be cancelled, kill them


            # 每个客户仅有一个订单，更新后删除原来的（目前看来是下一个周期的，即上一个周期没有卖掉）
            if len(kills) > 0:
                # if verbose : print('Kills: %s' % (kills))
                for kill in kills:
                    # if kill == self.agent_tid:
                    #     a=1
                    # if verbose : print('lastquote=%s' % traders[kill].lastquote)
                    if self.traders[kill].ttype == 'Agent' and self.training:
                    # if self.traders[kill].ttype == 'Agent':
                        self.traders[kill].profit_last = (self.traders[kill].range_limit[2]/2+self.traders[kill].range_limit[0]/2+self.traders[kill].range_limit[1]-self.traders[kill].lastquote.price)*0.3
                    #TODO 未成交惩罚,利用self.training标记

                    if self.traders[kill].lastquote is not None:
                        # if verbose : print('Killing order %s' % (str(traders[kill].lastquote)))
                        self.exchange.del_order(self.time, self.traders[kill].lastquote, verbose)
                    

                self.lob_update()

            # TODO Record
            # if self.time - self.time_record > 30:
            #     self.time_record = self.time
            #     bdump = open(f'data_temp_Agent{current_time}.csv', 'a')
            #     for t in self.traders:
            #         if self.traders[t].ttype=='Agent':
            #             bdump.write(f'{self.time},{self.traders[t].balance/self.time},\n')
            #     bdump.close()
                
            #     bdump = open(f'data_temp_PRSH{current_time}.csv', 'a')
            #     for t in self.traders:
            #         if self.traders[t].ttype=='PRSH':
            #             bdump.write(f'{self.time},{self.traders[t].balance/self.time},\n')
            #     bdump.close()
                
            #     bdump = open(f'data_temp_ZIP{current_time}.csv', 'a')
            #     for t in self.traders:
            #         if self.traders[t].ttype=='ZIP':
            #             bdump.write(f'{self.time},{self.traders[t].balance/self.time},\n')
            #     bdump.close()

                                
            #     bdump = open(f'data_temp_AA{current_time}.csv', 'a')
            #     for t in self.traders:
            #         if self.traders[t].ttype=='AA':
            #             bdump.write(f'{self.time},{self.traders[t].balance/self.time},\n')
            #     bdump.close()

            if flag:
                break #TODO 一轮游

            # get a limit-order quote (or None) from a randomly chosen trader
            # 每次抽取一个进行订单更新/发布，平均每秒更新一次
            tid = list(self.traders.keys())[random.randint(0, len(self.traders) - 1)]

            order = None
            if self.traders[tid].ttype == 'Agent':
                if self.agent_tid == -1:
                    self.agent_tid = tid
                if not len(self.traders[self.agent_tid].orders)<1:
                    return self._get_obs(), self._get_rew(), False

            else:
                order = self.traders[tid].getorder(self.time, self.time_left, self.exchange.publish_lob(self.time, self.lob_verbose))

            # 存在订单就更新 if verbose: print('Trader Quote: %s' % (order))

            if order is not None:
                
                #TODO record trader price
                # if self.traders[tid].ttype in ['Agent','AA','PRSH','ZIP']:
                #     bdump = open(f'data_temp_{self.traders[tid].ttype}{current_time}.csv', 'a')
                #     bdump.write(f'{self.time},{order.price},\n')
                #     bdump.close()

                # if order.otype == 'Ask' and order.price < self.traders[tid].orders[0].price:
                #     sys.exit('Bad ask')
                # if order.otype == 'Bid' and order.price > self.traders[tid].orders[0].price:
                #     sys.exit('Bad bid')
                # send order to exchange
                self.traders[tid].n_quotes = 1
                trade = self.exchange.process_order2(self.time, order, self.process_verbose)
                if trade is not None:
                    # trade occurred,
                    # so the counterparties update order lists and blotters
                    self.traders[trade['party1']].bookkeep(trade, order, self.bookkeep_verbose, self.time)
                    self.traders[trade['party2']].bookkeep(trade, order, self.bookkeep_verbose, self.time)
                    # 记录
                    # if dump_all:
                    #     trade_stats(sess_id, self.traders, tdump, self.time, self.exchange.publish_lob(time, lob_verbose))

                # traders respond to whatever happened
                lob = self.exchange.publish_lob(self.time, self.lob_verbose)
                for t in self.traders:
                    # NB respond just updates trader's internal variables
                    # doesn't alter the LOB, so processing each trader in
                    # sequence (rather than random/shuffle) isn't a problem
                    self.traders[t].respond(self.time, lob, trade, self.respond_verbose)
                
                self.lob_update()

            self.time = self.time + self.timestep

            # global data_fitness
            # if len(data_fitness)>num_k-1:
            #     data_fitness=dict()
            #     break 

        return self._get_obs(), self._get_rew(), True


    def step(self, action):
        
        action = action[0]

        #TODO 映射范围
        #price clip
        price = action*(self.maxprice-self.minprice)+self.minprice

        #price map
        # price = action*(self.maxprice-self.traders[self.agent_tid].orders[0].price)+self.traders[self.agent_tid].orders[0].price

        if not len(self.traders[self.agent_tid].orders)<1 and not self.training:
            if price < self.traders[self.agent_tid].orders[0].price:
                price = self.traders[self.agent_tid].orders[0].price

        self.agent_process_order(int(price))
        # print(price)

        obs, rew, done = self.continues()

        while len(self.traders[self.agent_tid].orders)<1:
            obs, rew, done = self.continues()
            
            if done == True:
                break
            #     self.reset()
        # print(done)

        return obs, rew/10, done, {}


    def reset(self):
        self.__init__(training=self.training,range2_input=self.range2_input,traders_per=self.traders_per)
        obs, rew, done = self.continues()
        return obs, rew, done, {}


if __name__ == "__main__":
    import time
    st = time.time()
    env = Agent2BSE(training=True)
    print(env.reset())
    for i in range(100):
        print(env.time,env.step([0.1]))
    (print(time.time()-st))

    # for i in range(10):
    #     a=np.random.randint(30)
    #     b=np.random.randint(30-a)
    #     c=30-a-b
    #     print(a,b,c)
