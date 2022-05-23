#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:44:29 2020

@author: Mikael
"""
from scipy import *
from matplotlib.pyplot import *
from numpy import *


class FAS:

    def __init__( self, x, N, U0, S_L, delta_t):
        self.x = x 
        self.delta_t = delta_t
        self.steps = N
        self.U0 = U0
        self.S_L = S_L

    
    def F(self,u):
        
        d = shape(u)[0]
        G = zeros((d))
        dx=1/d
        
        
        for i in range(d-1):
            G[i] = u[i+1]**2 - u[i]**2
        
        G[-1] = u[-1]**2 - u[-2]**2
        
        return   u + (self.delta_t/(2*dx))*G

    def Runge_Kutta(self , S_L , u_l ,iterations,dt_help,l):
        
        d = shape(u_l)[0]
        d = self.steps
        dx = 1/d
        ul_save = u_l.copy()
        dt = self.delta_t
        
        a_1 = 3.3
        
        
        c = 0.1/norm(u_l)
       
        dt_star = (c * dx) / ( dt )   
        
         
        
        
        for i in range(iterations):
             last_u_l = u_l     # if runge kutta made it worse we return the last value
             
             error_old = norm(self.F(u_l) - S_L) # check if  RK made it worse
             
             U1 = u_l - a_1*dt_star* (S_L - self.F(u_l ) )
             u_new = u_l + dt_star*(S_L - self.F(U1))
             
             u_l = u_new
             
             error_new = norm(self.F(u_l) - S_L) # Check if RK made it worse
             
             if norm(error_old) < norm( error_new ) :   
                 
               #print('whent up after', i, 'error_new', error_new)
               
               return last_u_l
        
        return u_l  
        
    
    
    def Vcycle(self,S_L, u_l, layer_max): # S_L is our right hand side and u_l is the guess we give to FAS that we will restrict and change
        original_b_RHS = S_L
        u_0 = u_l
        #F(u) - original_b_RHS == 0  --->  u = U^1
        first_error = norm( self.F(u_l) - S_L )/norm(S_L)
        print(first_error, "First error...")
        error = 1
        tol = 6.1*1e-10
        
        u_initial = u_l
        j = 0
        
        while norm(error) > tol:
            j +=1
            U_L_BAR = []
            list_of_P = []
            U_L_list = []
            
            list_of_SL = [S_L]
            
            
            u_l, list_of_P, list_of_SL, U_L_BAR, U_L_list = self.ForwardMultigrid(u_l , layer_max, U_L_BAR, list_of_P, list_of_SL,U_L_list)
            
            
            
            u_new , error = self.BackwardsMultigrid(u_l , layer_max, U_L_BAR , list_of_P, list_of_SL, original_b_RHS, U_L_list)
            
            u_l = u_new
            
           
            print(norm(error), "Error...",j,'iteration')
        
        plot_helper = np.zeros(2*len(u_0))
        plot_og = np.zeros(2*len(u_0))
        #print(len(plot_og) , len(plot_helper), "here")
        for i in range(len(u_0)):
            try:
                plot_helper[2*i] = u_l[i]
                plot_helper[2*i+1] = u_l[i]
                
                #plot_og[2*i] = u_0[i]
                #plot_og[2*i+1] = u_0[i]
            except:
                pass
        
        for i in range(len(u_0)-1):
            
            plot(self.x[i:i+2] ,plot_helper[2*i:2*i+2] )
            
            #plot( self.x[i:i+2] , plot_og[ 2*i:2*i+2] )
        
        
        print('FAS converged after' , j , 'iterations')
        
        return u_l
        
    



    def ForwardMultigrid(self, u_l, l_max, U_L_BAR, list_of_P , list_of_SL ,U_L_list):
        #print("Going down")
        
        help_c = norm(u_l)
        for L in range(l_max):
           
            l = l_max - L - 1
            
            if l == 0: #Coarsest
                iterations = 500    # this is the number of Runge kutta iterations we do on coarsest level to solve.
                U_L_list.append(u_l)
                u_0 = self.Runge_Kutta( S_L , u_l ,iterations,help_c, l )
                
                
                
                return u_0, list_of_P, list_of_SL, U_L_BAR, U_L_list
            
            else:
                
                S_L = list_of_SL[L]
                
                d = shape(S_L)[0]
                R = zeros((int(d/2),d))
                for i in range(int(d/2)):
                    try : 
                        R[i,i*2] = 0.5
                        R[i,i*2 +1 ] = 0.5
                        
                    except:
                          pass
    
                P = 2*R.T
                
                iterations = 3
                
                u_L_S = self.Runge_Kutta(S_L , u_l,iterations , help_c, l ) 
                
                U_L_list.append(u_l)
                
                
                r_L = S_L - self.F(u_L_S) #Computes residual
                
                u_L_R_S = dot(R,u_L_S)
                
                U_L_BAR.append(u_L_R_S)
                
                u_l_m1 = dot(R,u_l)
                u_l = u_l_m1
              
                S_L = dot(R,r_L) + self.F(u_L_R_S)        #Restricts residual
                
                
                list_of_P.append(P)
                list_of_SL.append(S_L)
    
    
    def BackwardsMultigrid(self, u_0 , l_max, U_L_BAR , list_of_P  , list_of_SL, original_b_RHS,U_L_list):
        
         
        U_L_list[-1] = u_0
        
        for l in range(l_max):
            
            
            if l == l_max - 2:    
                
                S_L = list_of_SL[0]
                
                Pu = np.dot(list_of_P[0], U_L_list[1] - U_L_BAR[0])     #Final prolongation
                
                u_new = U_L_list[0] + Pu #Correction
                
                u_new = self.Runge_Kutta( S_L , u_new , 10 , 30 , l_max - l  ) 
                
                
                test_residual = norm( self.F(u_new) - list_of_SL[0])/norm(original_b_RHS)
                
                
                return u_new , test_residual
            
            
            else:
                
                
                u_0 = U_L_list[-l-2] + np.dot(list_of_P[-l-1], U_L_list[-l-1] - U_L_BAR[-1-l] )    #Prolongation and correction U_L_BAR[-l-2] - u_0
                
                S_L = list_of_SL[-l-2]
                u_0 = self.Runge_Kutta( S_L , u_0 , 2, 1, l_max - l) 
                
                U_L_list[-l-2] = u_0
            
            
            

    


def CreateInitialValue(x):
    d = shape(x)[0]
    delta_x = 1/d
    extra_point = - delta_x
    f = lambda x : 2 + sin(pi*x)
    
    
    
    U_n = zeros((d))
    
    U_n[0] = (1/2)*( f(extra_point) + f(x[0]))
    
    
    for i in range(1,d):
        U_n[i] = (1/2)*( f(x[i-1]) + f(x[i]))
    
        
    plot_helper = zeros(2*d)
    
    return U_n


N = 64 ## 32768  16384   8192      4096     2048      1024      512      256       128     64     32    16    4    2
x_axis = linspace(0,2,N) 
U0 = CreateInitialValue(x_axis)
S_L = U0   



delta_t = 5*1e-4
delta_t = 0.9 / ( N * norm( S_L ) )   ###
layer = 3 # must be 2 or more
solution = FAS(x_axis , N, U0, S_L ,delta_t)

solution.Vcycle(U0,U0,layer)