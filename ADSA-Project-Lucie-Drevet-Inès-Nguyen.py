#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 10:06:54 2020

@author: luciedrevet
"""


############### VARIABLE #############

#model with the vents
model=[[1,2,2],
        [1,9,2.5],
        [1,10,2.7],
        [2,1,2],
        [2,3,5.5],
        [2,9,3.5],
        [2,10,2],
        [2,11,3],
        [3,2,5.5],
        [3,4,3.5],
        [3,8,4],
        [3,11,3],
        [3,13,3],
        [4,3,3.5],
        [4,5,3],
        [4,6,4.2],
        [4,12,2],
        [5,4,3],
        [5,6,3.5],
        [5,12,4],
        [6,4,4.2],
        [6,5,3.5],
        [6,12,2.5],
        [6,7,1.7],
        [6,8,4],
        [7,6,1.7],
        [7,8,2],
        [8,3,4],
        [8,6,4],
        [8,7,2],
        [8,9,5],
        [8,13,2.5],
        [8,14,2],
        [9,1,2.5],
        [9,2,3.5],
        [9,8,5],
        [9,10,2.2],
        [9,14,3],
        [10,1,2.7],
        [10,2,2],
        [10,9,2.2],
        [11,2,3],
        [11,3,3],
        [12,4,2],
        [12,5,4],
        [12,6,2.5],
        [13,3,3],
        [13,8,2.5],
        [14,8,2],
        [14,9,3]]
        
############### STEP 1 #############
import random

class Node:
    # we create the constructor of the class node 
    #our node will be composed by the player's number, his score
    #his total score (= mean), his left neighbor
    #his right neighbor and his height
    def __init__(self,player,score,mean,left=None,right=None):
        self.player=player
        self.score= score
        self.mean= mean
        self.left = left
        self.right = right
        self.height = 0
        
    #a method that calculate the height of a node 
    def set_height(self):
        # if the node has no children his height is 0
        if self.left is None and self.right is None:
            self.height = 0
        #otherwise, it's equal to the number of descendent he has +1
        elif self.left is None:
            self.height = self.right.height + 1
        elif self.right is None:
            self.height = self.left.height + 1
        else:
            self.height = max(self.right.height,self.left.height) + 1

    # a method that returns the balance factor
    # the balance factor is equal to
    # the height of the left and the right sub-trees
    def compute_balance(self):
        height_left = -1 if self.left is None else self.left.height
        height_right = -1 if self.right is None else self.right.height
        return height_left - height_right
    
    def insert_player(self,player,score,mean):
        # Insert the value
        
        if self.mean>=mean:
            if self.left is not None:
                self.left.insert_player(player,score,mean)
            else:
                self.left = Node(player,score,mean)
        else:
            if self.right is not None:
                self.right.insert_player(player,score,mean)
            else:
                self.right = Node(player,score,mean)

        # Update the height, important to compute a correct balance
        self.set_height()
        
        # balance should be between 1 and -1
        balance = self.compute_balance()
        if balance > 1:
            if self.left.mean >= mean:
                self.right_rotation()
                # every node moves one position to right 
            else:
                self.left.left_rotation()
                self.right_rotation()
                # first, every node moves one position to left 
                #then one position to right
        elif balance < -1:
            if self.right.mean >= mean:
                self.right.right_rotation()
                self.left_rotation()
                #every node moves one position to right
                #then one position to left 
            else:
                self.left_rotation()
                #every node moves one postition to left
                
    def right_rotation(self):
        # Rotation were every node moves one position to right
        temp_node = self.left.right # We save the right subtree of the left child of self
        right_node = Node(self.player, self.score,self.mean, temp_node,self.right) # We create the future right child of self
        self.player = self.left.player 
        self.left = self.left.left
        self.right = right_node

        # We update the height
        self.set_height() 
        self.right.set_height()

    def left_rotation(self):
        # Rotation were every node moves one position to left
        temp_node = self.right.left # We save the left subtree of the right child of self
        left_node = Node(self.player, self.score,self.mean, self.left,temp_node) # We create the future left child of self
        self.player = self.right.player
        self.right = self.right.right
        self.left = left_node

        # We update the height
        self.set_height()
        self.left.set_height()
        

        
    def display(self):
        lines, *_ = self._display_aux()
        for line in lines:
            print(line)

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.right is None and self.left is None:
            line = '%s' % self.player
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = '%s' % self.player
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = '%s' % self.player
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = '%s' % self.player
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2
  
# We create a function Score that will attribute a random score between 0 and 12 to a player    
def Score():
        score=random.randrange(0,13)   
        return score
 
# We create a function that will define which players are the impostors in the game
def Impostors(players):
    #we randomly implement the roles
    impostor1 = random.choice(players)
    impostor2 = random.choice(players)
    
    #we verify that the 2 impostors are different players
    if impostor1 == impostor2:
        Impostors(players)
        return impostor1, impostor2
    else:
        return impostor1, impostor2
 
# we associate a random pool to 10 different players
def RandomGame(previousPool):
    pool = random.randrange(1,11)
    # we assure ourself that the pool isn't already attributed
    if pool in previousPool:
        RandomGame(previousPool)
        return previousPool[-1]
    else:
        previousPool.append(pool)
        return pool
        
#This function returns tree lists that are sorted by the total score (mean of all the scores)
def inOrder(root): 
          # Set current to root of binary tree 
    current = root  
    stack = [] # initialize stack 
    players=[]
    scores=[]
    means=[]
      
    while True:          
        # Reach the left most Node of the current Node 
        if current is not None: 
              
            # Place pointer to a tree node on the stack  
            # before traversing the node's left subtree 
            stack.append(current)
            current = current.left  
  
          
        # BackTrack from the empty subtree and visit the Node 
        # at the top of the stack; however, if the stack is  
        # empty you are done 
        elif(stack): 
            current = stack.pop() 
            #print(current.player, end=" ") # Python 3 printing 
            players.append(current.player)
            scores.append(current.score)
            means.append(current.mean )
          
            # We have visited the node and its left  
            # subtree. Now, it's right subtree's turn 
            current = current.right  
  
        else: 
            break
       
    return players,scores,means
        
#We create our first AVL Tree
def First_avl_tree(value_list = []):
    root_node =  Node(value_list[0],0,0)
    for player in value_list[1:]:
        root_node.insert_player(player,0,0)
    return root_node

#We calculate the total score for each player at each game
def Mean(nb_game, saving , index):
    scoreTotal = 0
    for save in range(0,len(saving)-1, 2):
           #index is the player's number
           if saving[save] == index:
               scoreTotal = scoreTotal + saving[save +1]  
               
    #the total score is equal to all the scores divided the number of games
    mean = round(scoreTotal / nb_game)
    return mean
 
#We fill in our AVL Tree with the players number, his current score and his total score
def Database_game(nb_game, saving, value_list = []):
    score = Score()
    saving.append(value_list[0])
    saving.append(score)

    root_node =  Node(value_list[0], score, Mean(nb_game,saving ,value_list[0]))
    
    for player in value_list[1:]:
        saving.append(player)
        score = Score()
        saving.append(score)
        root_node.insert_player(player, score, Mean(nb_game,saving, player))
        
    return root_node  

#We create random games based on the database
# each game regroups the players by a batch of ten following their ranking until the final
def Games(jeux, ranking, excluded):
    players,scores,means= inOrder(jeux)
    # nbPool = number of pools
    nbPool=len(players)/10 +1
    
    #compteur will help us to separate the games into batch of 10
    compteur = 0
    previousPool = []
    
    #We separate the players into n group of 10
    while compteur < len(players):
        playersInPool = players[compteur:compteur+10]
        scoresInPool = scores[compteur:compteur+10]
       
        # a dictionary where the keys are the players and the values their scores
        playerScore = {player:score for (player,score) in zip(playersInPool,scoresInPool)}
        
        # a dictionary where the keys are the players and the values the total score of the players
        playerMean = {player:mean for (player,mean) in zip(players,means)}
        compteur+= 10
        
        ## We display the number of the pool for the 10 following players
        #If it's the 3 first games the pools are attributed randomly
        if ranking == 0: 
            pool = RandomGame(previousPool)
            print("POOL",pool,"\nThe 10 players are:", playersInPool)
            #print( "\nThe 10 players are in the pool :", pool,"\n\n", playersInPool, '\n')
            
        #If we are between the game 4 and 12, the pools are based on the ranking
        if ranking == 1:
            nbPool-=1
            print("POOL",nbPool,"\nThe 10 players are:", playersInPool)
     
        #If we are in the finals, the scores are reeinitalised and we display all the results
        if ranking == 2:   
            #we display the 10 players of the pool
            for player, mean in playerMean.items():
                if player <=9:
                    print('Player ',player, '  :  Total Score :', mean)
                if player == 100:
                    print('Player ',player, ':  Total Score :', mean)
                if player >=10 and player != 100:
                    print('Player ',player, ' :  Total Score :', mean)  
            
        #we affect the roles             
        impostor1, impostor2 = Impostors(playersInPool)
        print("The impostors were the players", impostor1, "and", impostor2,"!\n" )
     
    #we display the 10 excluded players of the pool and their total score
    if ranking ==1:
        print("The 10 players excluded from the following game are:")
        # we go through all the excluded players
        for eliminated in range(len (excluded)):
            # we display their total score by going through our dictionary playerMean
            for player, mean in playerMean.items():
               if player == excluded[eliminated] and excluded[eliminated] <=9:
                    print('Player excluded',player,'  :  Total Score :', mean)
                    
               if player == excluded[eliminated] and excluded[eliminated] == 100:
                    print('Player excluded',player, ':  Total Score :', mean)
                    
               if player == excluded[eliminated] and excluded[eliminated] >=10 and excluded[eliminated] != 100:
                    print('Player excluded',player, ' :  Total Score :', mean)  
        
                      
    return playerScore

#We create a function that will display the top 10 players for the 5 finals games
def TOP1O_and_podium(node):
    finalists, finalScore, finalMean = inOrder(node)
    print("\nHere are the 10 finalists : ")
    for i in range(len(finalists)):
        print("Player ", finalists[i])
    print("\nOn the podium we have :")
    podium=[9,8,7]
    for i, j in enumerate(podium):
        #i is equal to the fird, second or first place
        #j is equal to the player number
        print(str(i+1)+"e place for the player "+str(finalists[j]))
        
#This is our main function that will coordonate all the functions to create the tournament
def step1():
    #création first database where all scores = 0
    all_players=[i for i in range(1,101)]
    jeux=First_avl_tree(all_players)
  
    #saving = [player 1, score 1, player 2, score2, ..., player n, score n]
    saving = []
    nb_games = 0
    
    #nb_ranking is equal to the number of games played that are bases on rankings
    nb_ranking = 0
    for i in range(0,3):#the 3 random games
        nb_games += 1
        #random game 
        jeux=Database_game(nb_games,saving,all_players)#create a new database with actual scores
        print("\n-------------------------------------------")
        print("\nHere are the results of the game", nb_games)
        Games(jeux,0,0)
    n=101 #number of players at the begining
    excluded=[]
    while n>11:
        nb_games += 1
        nb_ranking += 1
        players_number=[]
        for i in range(1,101):
            if i not in excluded:
                players_number.append(i)
        #game based on ranking
        jeux=Database_game(nb_games,saving,players_number)
        players,scores,means= inOrder(jeux)
        ten_last=players[0:10]
        for i in range(len(ten_last)):
            excluded.append(ten_last[i])
        n-=10
        print("\n-------------------------------------------")
        print("Here are the results of the game", nb_games,"\nwith a number of", len(players_number),"players:\n")
        Games(jeux,1 ,ten_last) 
    final_player=[]
    reinitialised = []
    for i in range(1,101):
        if i not in excluded:
            final_player.append(i)
    print("\n-------------------------------------------\n                  FINALS")
    for i in range(0,5):
        #random game
        jeux=Database_game(i+1,reinitialised,final_player)
        print("\n-------------------------------------------")
        print("Here are the results of the game", i+1,"\n")
        Games(jeux,2, 0)
    TOP1O_and_podium(jeux) 

############### STEP 2 #############

def step2():
    
    ## Q4 ##

    neighbors = {}
    neighbors['1'] = ['2', '6']
    neighbors['2'] = ['1', '3', '7']
    neighbors['3'] = ['2', '4', '8']
    neighbors['4'] = ['3', '9']
    neighbors['5'] = ['7', '8']
    neighbors['6'] = ['1', '8', '9']
    neighbors['7'] = ['2', '5', '9']
    neighbors['8'] = ['3', '5', '6']
    neighbors['9'] = ['4', '6', '7']
    #dictionnary that define the "have seen" relationship

    colors = ['Red', 'Blue', 'Green']
    #color of our map

    colors_of_nodes = {}

    def promising(node, color):
        for neighbor in neighbors.get(node): #for loop to look at each neighbors of the node we are working one
            color_of_neighbor = colors_of_nodes.get(neighbor)#we  recuperate the color of the neighbors (if it doesn't have a color, noneobject is return)
            if color_of_neighbor == color: #if it has the same color than the color that we want to attribute it's not good
                return False #so we return false because the two nodes are linked and can't have the same color

        return True

    def get_color_for_node(node):
        for color in colors:#we  travel trough the different color possible
            if promising(node, color):#if neighbors doesn't have the same color then we can attribute it
                return color #we attribute the color

    def main():
        l=[]
        for node in neighbors.keys(): #we get each node one by one
            colors_of_nodes[node] = get_color_for_node(node)#attribution of color for the node we are working one
        for v in colors_of_nodes.values():#then we travel the dictionnary we created, more precisely the values
            l.append(v) #to add the color in another list

        imp=dict() #futur list of probable impostor
        i=[1,4,5] #our presume first probable impostor
        for j in i:
            imp[j]=list()#we create lists insidebthe dictionnary if the node we are looking at have several possibility of matchs (impostor)
            for i in range(len(l)):
                if l[j-1]==l[i] and (j-1!=i): #we look if the node and another node have the same color
                    imp[j].append(i+1)# if yes then they are not connected so it's a probabale pair of impostor
        print("")
        for key, value in imp.items():
            for v in value :
                print("Player ",key, ' can be associated with player ', v)


    main()

############### STEP 3 #############

def step3():
    #First model without vent : see matrix "model" at the beginning
    
    #Second model with vents
    model2=[[1,2,0],
        [1,9,0],
        [1,10,2.7],
        [2,1,0],
        [2,3,5.5],
        [2,9,3.5],
        [2,10,2],
        [2,11,3],
        [3,2,5.5],
        [3,4,3.5],
        [3,8,4],
        [3,11,3],
        [3,13,0],
        [4,3,3.5],
        [4,5,0],
        [4,6,4.2],
        [4,12,2],
        [5,4,0],
        [5,6,0],
        [5,12,4],
        [6,4,4.2],
        [6,5,0],
        [6,12,2.5],
        [6,7,1.7],
        [6,8,4],
        [7,6,1.7],
        [7,8,2],
        [8,3,4],
        [8,6,4],
        [8,9,5],
        [8,13,2.5],
        [8,14,2],
        [9,1,0],
        [9,2,3.5],
        [9,8,5],
        [9,10,2.2],
        [9,14,3],
        [10,1,2.7],
        [10,2,2],
        [10,9,2.2],
        [11,2,3],
        [11,3,3],
        [12,4,2],
        [12,5,4],
        [12,6,2.5],
        [13,3,0],
        [13,8,2.5],
        [14,8,2],
        [14,9,3],
        [10,11,0],
        [11,10,0],
        [10,14,0],
        [14,10,0],
        [11,14,0],
        [14,11,0]]
        
    ## Q3 ##

    V = 14 #number of node
    INF  = 99999#initiate value of distance between the node/room
  
    def floydWarshall(graph): #function for the distance between the rooms
  
    
        dist = list(map(lambda i : list(map(lambda j : j , i)) , graph) ) #we recuperate our matrix of distance
        #use of for loop to see all the matrix
        for k in range(V): 
            for i in range(V): 
                for j in range(V): 
                    if dist[i][j]>(dist[i][k]+ dist[k][j]): #we check if another way to go from a room to another is shorther that the first distance that we had
                        dist[i][j]=dist[i][k]+ dist[k][j] #if yes then we change it
                    
        return dist #we return the new matrix of distance
  
  
    def printSolution(dist): #function to print the distance between each room
        print ("Following matrix shows the shortest distances between every pair of vertices","\n")
        for i in range(V): 
            for j in range(V): 
                if(dist[i][j] == INF): 
                    print (i+1," -> ",j+1," = "+"%7s" %("INF"))
                else: 
                    print (i+1," -> ",j+1," = "+"%7d\t" %(dist[i][j]))
                if j == V-1: 
                    print ("") 
                    
    ## Q3&4 ##
  
    def find_impostor():#function to find the distance bewteen the rooms and then find the impostors
        graph1=[[INF for column in range(V)] for row in range(V)] #initiation of the distance matrix
        for i in range(len(model)):
            graph1[model[i][0]-1][model[i][1]-1]=model[i][2]#the we attribuate the distance that we measured on the map
            for j in range(V):
                graph1[j-1][j-1]=0#and this is for the distance between a room herself (which is zero)
        print("\n","Modele one","\n")
        printSolution(floydWarshall(graph1))#we print the first distance where there is not vent, for the crewmates
        print("   ")
        #and then same thing for the impostors
        graph2=[[INF for column in range(V)] for row in range(V)]
        for i in range(len(model2)):
            graph2[model2[i][0]-1][model2[i][1]-1]=model2[i][2]#we attribued the distance that we measured (include the one in plus)
            for j in range(V):
                graph2[j-1][j-1]=0
        print("Modele two","\n")
        printSolution(floydWarshall(graph2))#and we show the result
        #then we want to know which path is only for impostors
        print("Chemin de triche : ","\n")
        for j in range(V-1):
            #for that we compare the two matrix that we juste made : 
            if floydWarshall(graph1)[j+1][j]!=floydWarshall(graph2)[j+1][j]: #if there is a distance between two rooms thare not the same, that means that an impostor took a shorther path : a vent
                print (j+1," -> ",j+2," = "+"%7d\t" %(floydWarshall(graph2)[j+1][j])) #so we have the cheated path

    find_impostor() 

def step4():
    
    ## Q4 ##
    
    def hamilton(graph, start_v):#function that define a hamiltonian path from a start node
        size = len(graph)#we get the size of the graph that represent the map : here it's equal to the number of nodes : 14
        to_visit = [None, start_v]#like the name of the varibale said, it's the node that we will have to visit
        # if None we are -unvisiting- comming back and pop v
        path = []#our futur hamiltonian path
        while(to_visit):
            v = to_visit.pop()
            if v : 
                path.append(v)#we had the node that we visited to the path
                if len(path) == size:#if the path is complete, that we visited all the map then it's over
                    break
                for x in set(graph[v])-set(path):# for loop to visit each neighbors of the node we had to the path
                    to_visit.append(None) # out, we had a noneobject to be able to delete a node from the path if it's not concluent
                    to_visit.append(x) # in, we had the neighbors, to work on the neighbors of the neighbors after that
            else: # if None we are comming back and pop v
                path.pop()
        return path #and we return the path
                
    #matrix that shows the connection between each node/room
    G = {1:[2,9,10], 2:[1,3,9,10,11], 3:[2,4,8,11,13], 4:[3,5,6,12], 5:[4,6,12], 6:[4,5,7,8,12], 7:[6,8], 8:[3,6,9,13,14], 9:[1,2,8,10,14], 10:[1,2,9], 11:[2,3], 12:[4,5,6], 13:[3,8], 14:[8,9]}
    dist=[]#matrix that will contain each path that exist in our map
    for i in range(1,14):
        if len(hamilton(G, i))!=0:
            dist.append(hamilton(G, i))#we had the hamiltonian path from all starting node that exists
            
    dist_tot=[0 for i in range(len(dist))]#matrix that will conatin the length of each path
    for i in range(len(dist)):
        for j in range(13):
            for k in range(len(model)):
                if dist[i][j]==model[k][0] and dist[i][j+1]==model[k][1]:#we calculate the length by using the matrix that we juste created and the model of the map we created by taking measure on the map
                    dist_tot[i]+=model[k][2]

    b=min(dist_tot)# we recuperate the smallest one
    #And finally we print it
    print("\n","Voici le chemin le plus rapide, il dure",b," sec : ","\n")
    for i in range(len(dist_tot)):
        if dist_tot[i]==b:
            for j in range(13):
                print((dist[i][j]),"->",(dist[i][j+1]))
            break;
            
################# MAIN ###########            

print("Welcome in the amoung us game ")
answer=999
from sys import exit

while answer!="0":
    print("\n","Which step do you want to look at ?","\n")
    print("1 : step1")
    print("2 : step2")
    print("3 : step3")
    print("4 : step4")
    print("0 : quit the session")
    answer=input("Answer : ")
    if answer=="1":
        step1()
    if answer=="2":
        step2()
    if answer=="3":
        step3()
    if answer=="4":
        step4()
    if answer =="0":
        print("See you soon :) ")
        exit()
