#!/usr/bin/env python
import rospy
from math import *
import tf 
import rosbag
import numpy as np
import rospkg

#Conversions from continuos to discrete and vice versa
def continuosToDiscrete_dist(valueInMeters):
   discrete=int((valueInMeters*100)/20)-1
   return discrete

def discreteToContinuos_dist(valueInCell):
   continuos=((valueInCell*20)/100)+0.1
   return continuos

def continuosToDiscrete_angles(valueInDegrees):
   discrete=int(valueInDegrees/(discretization_size+1))
   return discrete

def discreteToContinuos_angles(valueInCell):
   continuos=-135+valueInCell*discretization_size
   return continuos
      
#a collective function which will convert discrete x,y and theta value to continuos value
def discreteToContinuos_main(x,y,theta):
   x_cont=discreteToContinuos_dist(x)
   y_cont=discreteToContinuos_dist(y)
   theta_cont=discreteToContinuos_angles(theta)
   return x_cont,y_cont,theta_cont


global o_grid
global q
global cmTags
global prediction
prediction=[]
global update
update=[]
#initialize occupancy grid with zeros first
o_grid=np.zeros((35,35,4))
#initialize discretization size to 90(this will divide the 360 degrees into 4 discrete values)
discretization_size=90
# init_angle=200.52
# cell_angle=continuosToDiscrete_angles(init_angle)
# print(cell_angle)
# o_grid[11,27,cell_angle]=1


#create initial gaussian distribution where 12,28,3 is the initial value so 0.8 is assgned to that and then its neighbours are assigned sum of 0.2
def get_neighbours(i,j,offset,value):
   for k in range(4):
      o_grid[i-offset,j,k]=value
      o_grid[i,j-offset,k]=value
      o_grid[i,j+offset,k]=value
      o_grid[i+offset,j,k]=value
      o_grid[i-offset,j-offset,k]=value
      o_grid[i-offset,j+offset,k]=value
      o_grid[i+offset,j-offset,k]=value
      o_grid[i+offset,j+offset,k]=value

#for initial gaussian distribution
offset_dict={0:0.8,1:(0.2/8)}
for key,value in offset_dict.items():
   get_neighbours(11,27,key,value)


#initialize the landmark 
mTags=[[1.25, 5.25],[1.25, 3.25],[1.25, 1.25],[4.25, 1.25],[4.25, 3.25],[4.25, 5.25]]


#read rosbag
rospack=rospkg.RosPack()
path=rospack.get_path('ros_pa3')
path=path+'/grid.bag'
bag = rosbag.Bag(path)
messages=[]
for topic, msg, t in bag.read_messages(topics=[ 'Movements','Observations']):
   #print(topic,msg,"/n")
   
   messages.append(msg)
bag.close()

#collect movements and measurements data, change all quaternions to eulers, convert all angles between -360 and 360
measurements_data=[]
movement_data=[]
for message in messages:
   #print(message)
   if message.timeTag%2 == 0:
      #print("sensor")
      current_measurement={}
      bearing_quaternion=(
         message.bearing.x,
         message.bearing.y,
         message.bearing.z,
         message.bearing.w
      )
      bearing_euler=tf.transformations.euler_from_quaternion(bearing_quaternion)
      temp=np.degrees(bearing_euler[2])
      if temp > 180:
         temp = temp_2 - 360
      elif temp < -180:
         temp = temp + 360	
      current_measurement['bearing']=temp
      current_measurement['tagNum']=message.tagNum
      current_measurement['range']=message.range
      measurements_data.append(current_measurement)
   else:
      #print(message.rotation1)
      current_movement={}
      rotation1_quaternion=(
         message.rotation1.x,
         message.rotation1.y,
         message.rotation1.z,
         message.rotation1.w
      )
      rotation1_euler=tf.transformations.euler_from_quaternion(rotation1_quaternion)
      rotation2_quaternion=(
         message.rotation2.x,
         message.rotation2.y,
         message.rotation2.z,
         message.rotation2.w
      )
      rotation2_euler=tf.transformations.euler_from_quaternion(rotation2_quaternion)
      temp_1=np.degrees(rotation1_euler[2])
      if temp_1 > 180:
         temp_1 = temp_1 - 360
      elif temp_1 < -180:
         temp_1 = temp_1 + 360
      current_movement['rotation1_euler']=temp_1
      current_movement['translation']=message.translation
      temp_2=np.degrees(rotation2_euler[2])
      if temp_2 > 180:
         temp_2 = temp_2 - 360
      elif temp_2 < -180:
         temp_2 = temp_2 + 360	
      current_movement['rotation2_euler']=temp_2
      movement_data.append(current_movement)

#odomentry motion model
def odom_motion_model(data):
   #print(data,data['rotation1_euler'])
   global o_grid
   global update
   global q
   #create temporary array and copy the occupancy grid to it so that it will keep record of previous values
   q = o_grid
   o_grid=np.zeros((35,35,4))
   #used for nomalization
   sum_p=0
   for i in range(35):
      for j in range(35):
         for k in range(4):
            #set threshold  to cut out the values with minimal probablities
            if q[i,j,k] < 0.0001:
               continue
            for a in range(35):
               for b in range(35):
                  for c in range(4):
                     #convert all occupancy grid values to it's corresponding continuos value for calculating the gaussian
                     i_c,j_c,k_c=discreteToContinuos_main(i,j,k)
                     a_c,b_c,c_c=discreteToContinuos_main(a,b,c)
                     #calculate rotation and translation
                     x_diff=i_c-a_c
                     y_diff=j_c-b_c
                     ang=degrees(np.arctan2(y_diff,x_diff))
                     trans_bar=np.sqrt((x_diff*x_diff) +(y_diff*y_diff))
                     rot1_bar=k_c-ang
                     rot2_bar=ang-c_c

                     #handle angles
                     if rot1_bar > 180:
                        rot1_bar = rot1_bar - 360
                     elif rot1_bar < -180:
                        rot1_bar = rot1_bar + 360		
                     if rot2_bar > 180:
                        rot2_bar = rot2_bar - 360
                     elif rot2_bar < -180:
                        rot2_bar = rot2_bar + 360

                     #find mean (mu) and standar deviation(sd_rot and sd_trans), then calculate gaussian
                     mu_rot1=rot1_bar-data['rotation1_euler']
                     sd_rot=45
                     p_rot1 = (1.0/(np.sqrt(2*np.pi)*sd_rot))*np.power(np.e,-1.0*((mu_rot1**2)/(2.0*sd_rot**2)))
                     mu_rot2=rot2_bar-data['rotation2_euler']
                     p_rot2 = (1.0/(np.sqrt(2*np.pi)*sd_rot))*np.power(np.e,-1.0*((mu_rot2**2)/(2.0*sd_rot**2)))
                     mu_trans=trans_bar-data['translation']
                     sd_trans=10
                     p_trans = (1.0/(np.sqrt(2*np.pi)*sd_trans))*np.power(np.e,-1.0*((mu_trans**2)/(2.0*sd_trans**2)))

                     #find joint probablities
                     final_prob=q[i,j,k]*p_rot1*p_rot2*p_trans
                     #print(final_prob)
                     #update to occupancy grid
                     o_grid[i,j,k]=o_grid[i,j,k]+final_prob
                     #find sum for normalizing
                     sum_p=sum_p+o_grid[i,j,k]
   #normalize the occupancy grid
   o_grid = o_grid / sum_p   
   #get the indices with maximum probablity value    
   print(np.unravel_index(o_grid.argmax(), o_grid.shape))
   update.append(np.unravel_index(o_grid.argmax(), o_grid.shape))


#sensor model z-measurements
def sense_model(z):
   global o_grid
   global cmTags
   global prediction
   global q
   #create a temp array to store previous values
   q = o_grid
   o_grid=np.zeros((35,35,4))
   #get current landmark index
   tag=mTags[z['tagNum']]
   #print(tag,o_grid[11,27,3])
   #initialize sum for normalization
   sum_p=0
   for i in range(35):
      for j in range(35):
         for k in range(4):
            #convert all occupancy grid values to it's corresponding continuos value for calculating the gaussian
            i_c,j_c,k_c=discreteToContinuos_main(i,j,k)
            #calculate rotation and translation
            x_diff=i_c-tag[0]
            y_diff=j_c-tag[1]
            rot_bar=degrees(np.arctan2(y_diff,x_diff)-radians(k_c))
            ang=degrees(np.arctan2(y_diff,x_diff))
            rot_bar=ang-k_c
            #handle angles
            if rot_bar > 180:
               rot_bar = rot_bar - 360
            elif rot_bar < -180:
               rot_bar = rot_bar + 360
            trans_bar=np.sqrt((x_diff*x_diff) +(y_diff*y_diff))

            #find mean (mu) and standar deviation(sd_rot and sd_trans), then calculate gaussian
            mu_rot=z['bearing']-rot_bar
            sd_rot=45
            p_rot = (1.0/(np.sqrt(2*np.pi)*sd_rot))*np.power(np.e,-1.0*((mu_rot**2)/(2.0*sd_rot**2)))
            mu_trans=z['range']-trans_bar
            sd_trans=10
            p_trans = (1.0/(np.sqrt(2*np.pi)*sd_trans))*np.power(np.e,-1.0*((mu_trans**2)/(2.0*sd_trans**2)))
            #calculate joint probablities
            final_prob=q[i,j,k]*p_rot*p_trans
            #print(final_prob)
            sum_p=sum_p+final_prob
            #update occupancy grid
            o_grid[i,j,k]=o_grid[i,j,k]+final_prob


   #normalize occupancy grid
   o_grid = o_grid / sum_p     
   #get indices with max probablity     
   print(np.unravel_index(o_grid.argmax(), o_grid.shape))
   prediction.append(np.unravel_index(o_grid.argmax(), o_grid.shape))


#bayes filter function
def bayes_filter():
   global prediction
   global update
   #loop through rosbag and implement movement and sense
   for k in range(len(movement_data)):   
      odom_motion_model(movement_data[k])
      sense_model(measurements_data[k])
   f = open("estimation.txt", "a+")
   for k in range(len(prediction)):
      prediction_x,prediction_y,prediction_theta=discreteToContinuos_main(prediction[k][0],prediction[k][1],prediction[k][2])
      update_x,update_y,update_theta=discreteToContinuos_main(update[k][0],update[k][1],update[k][2])
      str1="P: ("+str(prediction_x)+","+str(prediction_y)+","+str(prediction_theta)+")\n"
      str2="U: ("+str(update_x)+","+str(update_y)+","+str(update_theta)+")\n"
      f.write(str1)
      f.write(str2)
   f.close()
   print("End")

#call bayes filter
bayes_filter()
def listener():

   rospy.init_node('sensor_sub', anonymous=False)
   
    # spin() simply keeps python from exiting until this node is stopped
   rospy.spin()

if __name__ == '__main__':
    listener()
