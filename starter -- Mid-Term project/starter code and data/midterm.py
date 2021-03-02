'''
CSC381: Building a simple Recommender System

The final code package is a collaborative programming effort between the
CSC381 student(s) named below, the class instructor (Carlos Seminario), and
source code from Programming Collective Intelligence, Segaran 2007.
This code is for academic use/purposes only.

CSC381 Programmer/Researcher: Jake Carver

'''

import os
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import statistics 
from scipy import stats
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
import pickle
import pandas as pd
import sys
import copy


def sim_pearson(prefs,p1,p2):
    '''
        Calculate Pearson Correlation similarity 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2
        
        Returns:
        -- Pearson Correlation similarity as a float
        
    '''
    names = []
    for name in list(prefs[p1].keys() & prefs[p2].keys()):
        names.append(name)
    lista = []
    listb = []
    for i in names:
        lista.append(prefs[p1][i])
        listb.append(prefs[p2][i])
       
    n = len(names)
    if len(names)==0:
        return 0
    x = []
    y = []
    for i in range (len(names)):
        x.append(lista[i])
        y.append(listb[i])
    sum_x = float(sum(x))
    sum_y = float(sum(y))
    sum_x_sq = sum(xi*xi for xi in x)
    sum_y_sq = sum(yi*yi for yi in y)
    tot_sum = sum(xi*yi for xi, yi in zip(x, y))
    top = tot_sum - (sum_x * sum_y/n)
    bottom = sqrt((sum_x_sq - sum_x**2 / n) * (sum_y_sq - sum_y**2 / n))
    if bottom == 0: 
        return 0
    return top / bottom
    
    
 
    
    ##
    ## REQUIREMENT! For this function, calculate the pearson correlation
    ## "longhand", i.e, calc both numerator and denominator as indicated in the
    ## formula. You can use sqrt (from math module), and average from numpy.
    ## Look at the sim_distance() function for ideas.
    ##



def sim_distance(prefs,person1,person2):
    '''
        Calculate Euclidean distance similarity 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2
        
        Returns:
        -- Euclidean distance similarity as a float
        
    '''
    
    # Get the list of shared_items
    si={}
    for item in prefs[person1]: 
        if item in prefs[person2]: 
            si[item]=1
    
    # if they have no ratings in common, return 0
    if len(si)==0: 
        return 0
    
    # Add up the squares of all the differences
    sum_of_squares=sum([pow(prefs[person1][item]-prefs[person2][item],2) 
                        for item in prefs[person1] if item in prefs[person2]])
    
    sum_of_squares = 0
    for item in prefs[person1]:
        if item in prefs[person2]:
            #print(item, prefs[person1][item], prefs[person2][item])
            sq = pow(prefs[person1][item]-prefs[person2][item],2)
            #print (sq)
            sum_of_squares += sq
        
    return 1/(1+sqrt(sum_of_squares))


def calculateSimilarUsers(prefs,n=100,similarity=sim_pearson):

    '''
        Creates a dictionary of items showing which other items they are most 
        similar to. 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- n: number of similar matches for topMatches() to return
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A dictionary with a similarity matrix
        
    '''     
    result={}
    c=0
    for user in prefs:
      # Status updates for larger datasets
        c+=1
        if c%100==0: 
            print ("%d / %d") % (c,len(prefs))
            
        # Find the most similar items to this one
        scores=topMatches(prefs,user,similarity,n=n)
        result[user]=scores
    return result

def getRecommendationsSim(matrix,person,similarity=sim_pearson):
    totals={}
    simSums={}
    for other in matrix:
      # don't compare me to myself
        if other==person: 
            continue
        sim=similarity(matrix,person,other)
    
        # ignore scores of zero or lower
        if sim<=0: continue
        for item in matrix[other]:
            
            # only score movies I haven't seen yet
            if item not in matrix[person] or matrix[person][item]==0:
                # Similarity * Score
                totals.setdefault(item,0)
                totals[item]+=matrix[other][item]*sim
                # Sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+=sim
  
    # Create the normalized list
    rankings=[(total/simSums[item],item) for item,total in totals.items()]
  
    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings
def getRecommendedItems(prefs,itemMatch,user):
    '''
        Calculates recommendations for a given user 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''    
    userRatings=prefs[user]
    scores={}
    totalSim={}
    # Loop over items rated by this user
    for (item,rating) in userRatings.items( ):
  
      # Loop over items similar to this one
        for (similarity,item2) in itemMatch[item]:
    
            # Ignore if this user has already rated this item
            if item2 in userRatings: continue
            # ignore scores of zero or lower
            if similarity<=0: continue            
            # Weighted sum of rating times similarity
            scores.setdefault(item2,0)
            scores[item2]+=similarity*rating
            # Sum of all the similarities
            totalSim.setdefault(item2,0)
            totalSim[item2]+=similarity
  
    # Divide each total score by total weighting to get an average

    rankings=[(score/totalSim[item],item) for item,score in scores.items( )]    
  
    # Return the rankings from highest to lowest
    rankings.sort( )
    rankings.reverse( )
    return rankings
           
def get_all_II_recs(prefs, itemsim, sim_method, num_users=10, top_N=5):
    ''' 
    Print item-based CF recommendations for all users in dataset

    Parameters
    -- prefs: U-I matrix (nested dictionary)
    -- itemsim: item-item similarity matrix (nested dictionary)
    -- sim_method: name of similarity method used to calc sim matrix (string)
    -- num_users: max number of users to print (integer, default = 10)
    -- top_N: max number of recommendations to print per user (integer, default = 5)

    Returns: None
    
    '''
    for i in list(prefs.keys()):
        print('Item-based CF recs for '+i+' '+sim_method, getRecommendedItems(prefs,itemsim,i))

def loo_cv_sim(prefs,  sim, algo, sim_matrix):
    """
    Leave-One_Out Evaluation: evaluates recommender system ACCURACY
     
     Parameters:
         prefs dataset: critics, etc.
	 metric: MSE, or MAE, or RMSE
	 sim: distance, pearson, etc.
	 algo: user-based recommender, item-based recommender, etc.
         sim_matrix: pre-computed similarity matrix
	 
    Returns:
         error_total: MSE, or MAE, or RMSE totals for this set of conditions
	 error_list: list of actual-predicted differences
    """
    #getRecommendedItems(prefs,itemMatch,user)
    #loo_cv(prefs, metric, sim, algo)
    
    true_list = []
    pred_list = []
    error_list = []
    
    #Start
    for i in list(prefs.keys()):
        for out in list(sim_matrix.keys()):
            sumProd = 0
            sumSim = 0
            if out in list(prefs[i]):
                newPrefs = copy.deepcopy(prefs)
                del newPrefs[i][out]
                recList = algo(newPrefs,sim_matrix,i)
                
                for j in recList:
                    if j[1] == out:
                                #real.append(prefs[i][k[1]])
                                #suggested.append(k[0])
                                #print("Similarity: ",k[0])
                                #print("Rating: ",prefs[i][k[1]])
                        
                        try:
                            true_list.append(prefs[i][j[1]])
                            pred_list.append(j[0])
                            
                            error = (prefs[i][j[1]]-j[0])**2
                            error_list.append(error)
                            true_list.append(prefs[i][j[1]])
                            pred_list.append(j[0])
                            print("User: "+i+", Item: "+j[1]+ " Prediction: ", j[0]," Actual: ", prefs[i][j[1]]," Error: ", error)
                            
                        except:
                             print("User: "+i+", Item: "+j[1]+ ", No Prediction Available")
    #End
    
    '''
    for i in list(prefs.keys()):
        for j in list(sim_matrix.keys()):
            if j in list(prefs[i]):
                sumProd = 0
                sumSim = 0
                #real = []
                #suggested = []
                for k in list(sim_matrix[j]):
                    if k[1] in prefs[i] and k[0] > 0:
                        #real.append(prefs[i][k[1]])
                        #suggested.append(k[0])
                        #print("Similarity: ",k[0])
                        #print("Rating: ",prefs[i][k[1]])
                        sumProd += prefs[i][k[1]]*k[0]
                        #sumProd += prefs[i][j]*k[0]
                        sumSim +=abs(k[0])
                        
                try:
                    total = sumProd / sumSim
                    
                    error = (prefs[i][j]-total)**2
                    error_list.append(error)
                    true_list.append(prefs[i][j])
                    pred_list.append(total)
                    print("User: "+i+", Item: "+j+ " Prediction: ", total," Actual: ", prefs[i][j]," Error: ", error)
                    
                except:
                     print("User: "+i+", Item: "+j+ ", No Prediction Available")
            
            
            
            except:
                print (key+' NaN')
            '''
    
            
    '''  
    loo = LeaveOneOut()
    loo.get_n_splits(prefs)
    '''
    
    print('MSE: ', mean_squared_error(true_list, pred_list))
    print('MAE: ',mean_absolute_error(true_list, pred_list))
    print("RMSE: ", mean_squared_error(true_list, pred_list, squared=False))
    
    return error_list


def topMatches(prefs,person,similarity=sim_pearson, n=5):
    '''
        Returns the best matches for person from the prefs dictionary

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        -- n: number of matches to find/return (5 is default)
        
        Returns:
        -- A list of similar matches with 0 or more tuples, 
           each tuple contains (similarity, item name).
           List is sorted, high to low, by similarity.
           An empty list is returned when no matches have been calc'd.
        
    '''     
    scores=[(similarity(prefs,person,other),other) 
                    for other in prefs if other!=person]
    scores.sort()
    scores.reverse()
    return scores[0:n]

def transformPrefs(prefs):
    '''
        Transposes U-I matrix (prefs dictionary) 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        
        Returns:
        -- A transposed U-I matrix, i.e., if prefs was a U-I matrix, 
           this function returns an I-U matrix
        
    '''     
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})
            # Flip item and person
            result[item][person]=prefs[person][item]
    return result

def calculateSimilarItems(prefs,n=100,similarity=sim_pearson):

    '''
        Creates a dictionary of items showing which other items they are most 
        similar to. 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- n: number of similar matches for topMatches() to return
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A dictionary with a similarity matrix
        
    '''     
    result={}
    # Invert the preference matrix to be item-centric
    itemPrefs=transformPrefs(prefs)
    c=0
    for item in itemPrefs:
      # Status updates for larger datasets
        c+=1
        if c%100==0: 
            print ("%d / %d" % (c,len(itemPrefs)))
            
        # Find the most similar items to this one
        scores=topMatches(itemPrefs,item,similarity,n=n)
        result[item]=scores
    return result
                          

def from_file_to_dict(path, datafile, itemfile):
    ''' Load user-item matrix from specified file 
        
        Parameters:
        -- path: directory path to datafile and itemfile
        -- datafile: delimited file containing userid, itemid, rating
        -- itemfile: delimited file that maps itemid to item name
        
        Returns:
        -- prefs: a nested dictionary containing item ratings for each user
    
    '''
    
    # Get movie titles, place into movies dictionary indexed by itemID
    movies={}
    try:
        with open (path + '/' + itemfile) as myfile: 
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (id,title)=line.split('|')[0:2]
                movies[id]=title.strip()
    
    # Error processing
    except UnicodeDecodeError as ex:
        print (ex)
        print (len(movies), line, id, title)
        return {}
    except Exception as ex:
        print (ex)
        print (len(movies))
        return {}
    
    # Load data into a nested dictionary
    prefs={}
    for line in open(path+'/'+ datafile):
        #print(line, line.split('\t')) #debug
        (user,movieid,rating,ts)=line.split('\t')
        user = user.strip() # remove spaces
        movieid = movieid.strip() # remove spaces
        prefs.setdefault(user,{}) # make it a nested dicitonary
        prefs[user][movies[movieid]]=float(rating)
    
    #return a dictionary of preferences
    return prefs

def data_stats(prefs, filename):
    ''' Computes/prints descriptive analytics:
        -- Total number of users, items, ratings
        -- Overall average rating, standard dev (all users, all items)
        -- Average item rating, standard dev (all users)
        -- Average user rating, standard dev (all items)
        -- Matrix ratings sparsity
        -- Ratings distribution histogram (all users, all items)

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- filename: string containing name of file being analyzed
        
        Returns:
        -- None

    '''
    print('Total Users: '+str(len(prefs)))
    filmCount = {}
    filmRatings = {}
    ratingList = []
    for i in prefs:
        for j in prefs[i]:
            if j not in filmCount:
                filmCount[j] = 1
            else:
                filmCount[j]+=1
            if j not in filmRatings:
                filmRatings[j] = [prefs[i][j]]
            else:
                filmRatings[j].append(prefs[i][j])
                
            ratingList.append(prefs[i][j])
    averageRating = sum(ratingList)/len(ratingList)          
    print('Total Films: '+str(len(filmCount)))
    print('Total Ratings: '+str(len(ratingList)))
    print('Average Rating: '+str(averageRating))
    print('Overall Rating SD: '+str(statistics.stdev(ratingList)))
    print('STATS FOR EACH ITEM')
    for i in filmRatings:
        print(str(i)) 
        print('Average: '+str(sum(filmRatings[i])/len(filmRatings[i])))
        print('SD: '+str(statistics.stdev(filmRatings[i])))
    print('STATS FOR EACH USER')
    for i in prefs:
        userRatings=[]
        for j in prefs[i]:
            userRatings.append(prefs[i][j])
        print(str(i)) 
        print('Average: '+str(sum(userRatings)/len(userRatings)))
        print('SD: '+str(statistics.stdev(userRatings)))
    histList = []
    
    for i in prefs:
        for j in list(prefs[i].values()):
            histList.append(j)
    print (histList)  
    plt.hist(histList, bins = [1,2,3,4,5])
    plt.show()

def popular_items(prefs, filename):
    ''' Computes/prints popular items analytics    
        -- popular items: most rated (sorted by # ratings)
        -- popular items: highest rated (sorted by avg rating)
        -- popular items: highest rated items that have at least a 
                          "threshold" number of ratings
        
        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- filename: string containing name of file being analyzed
        
        Returns:
        -- None

    '''
    filmCount = {}
    filmRatings = {}
    for i in prefs:
        for j in prefs[i]:
            if j not in filmCount:
                filmCount[j] = 1
            else:
                filmCount[j]+=1
            if j not in filmRatings:
                filmRatings[j] = [prefs[i][j]]
            else:
                filmRatings[j].append(prefs[i][j])
    print('Most Rated')    
    for w in sorted(filmCount, key=filmCount.get, reverse=True):
        print(w, filmCount[w])
    
    highestRated = {}
    threshRated = {}
    for i in filmRatings:
        highestRated[i] = sum(filmRatings[i])/len(filmRatings[i])
        
        #Threshold set arbitrarily to 5
        if len(filmRatings[i]) > 5:
            threshRated[i]= sum(filmRatings[i])/len(filmRatings[i])
    
    print('Highest Rated')      
    for w in sorted(highestRated, key=highestRated.get, reverse=True):
        print(w, highestRated[w])   
    print('Highest Rated (Minimum 5 Ratings)')          
    for w in sorted(threshRated, key=threshRated.get, reverse=True):
        print(w, threshRated[w])
                
        
def getRecommendations(prefs,person,similarity=sim_pearson):
    '''
        Calculates recommendations for a given user 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''
    
    totals={}
    simSums={}
    for other in prefs:
      # don't compare me to myself
        if other==person: 
            continue
        sim=similarity(prefs,person,other)
    
        # ignore scores of zero or lower
        if sim<=0: continue
        for item in prefs[other]:
            
            # only score movies I haven't seen yet
            if item not in prefs[person] or prefs[person][item]==0:
                # Similarity * Score
                totals.setdefault(item,0)
                totals[item]+=prefs[other][item]*sim
                # Sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+=sim
  
    # Create the normalized list
    rankings=[(total/simSums[item],item) for item,total in totals.items()]
  
    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings



def loo_cv(prefs, metric, sim, algo):
    """
    Leave_One_Out Evaluation: evaluates recommender system ACCURACY
     
     Parameters:
         prefs dataset: critics, ml-100K, etc.
         metric: MSE, MAE, RMSE, etc.
         sim: distance, pearson, etc.
         algo: user-based recommender, item-based recommender, etc.
     
    Returns:
         error_total: MSE, MAE, RMSE totals for this set of conditions
         error_list: list of actual-predicted differences
    
    
    Algo Pseudocode ..
    Create a temp copy of prefs
    
    For each user in temp copy of prefs:
      for each item in each user's profile:
          delete this item
          get recommendation (aka prediction) list
          restore this item
          if there is a recommendation for this item in the list returned
              calc error, save into error list
          otherwise, continue
      
    return mean error, error list
    """
    newPrefs = prefs
    true_list = []
    pred_list = []
    error_list = []
    for i in list(prefs.keys()):
        for j in list(prefs[i].keys()):
            key = j
            value = newPrefs[i][key]
            
            newPrefs[i].pop(j, None)
            #try:
            recs = getRecommendations(newPrefs, i, similarity=sim)
            for k in recs:
                if k[1] == key:
                    true_list.append(value)
                    pred_list.append(k[0])
                    error_list.append((value-k[0])**2)
                    print (k[1]+' '+str((value-k[0])**2))
            '''
            except:
                print (key+' NaN')
            '''
            newPrefs[i][key]=value
    
            
    '''  
    loo = LeaveOneOut()
    loo.get_n_splits(prefs)
    '''
    error = metric(true_list, pred_list)
    print('Error: ',error)
    return error, error_list


def main():
    ''' User interface for Python console '''
    
    # Load critics dict from file
    path = os.getcwd() # this gets the current working directory
                       # you can customize path for your own computer here
    print('\npath: %s' % path) # debug
    done = False
    prefs = {}
    
    
    
    while not done: 
        print()
        # Start a simple dialog
        file_io = input('R(ead) critics data from file?, '
                        'P(rint) the U-I matrix?, '
                        'V(alidate) the dictionary?, '
                        'S(tats) print?,'
                        'D(istance)?,'
                        'PC(earson Correlation) critics data?, '
                        'U(ser) reccomendations?, '
                        'LCV Leave One Out Evaluation?, '
                        'Sim(ilarity matrix) calc for Item-based recommender?, '
                        'I(tem-based CF Recommendations)?, '
                        'LCVSIM(eave one out cross-validation)? ')
        
        if file_io == 'PC' or file_io == 'pc':
            print()
            if len(prefs) > 0:             
                print ('Example:')
                print ('Pearson sim Lisa & Gene:', sim_pearson(prefs, 'Lisa', 'Gene')) # 0.39605901719066977
                print()
                
                print('Pearson for all users:')
                # Calc Pearson for all users
                
                ## add some code here to calc User-User Pearson Correlation similarities 
                ## for all users or add a new function to do this
                
                names = list(prefs.keys())
                for i in range (len(names)):
                    for j in range (i+1, len(names)):
                        if i != j:
                            print ('Distance sim '+names[i]+' and '+names[j]+' '+str(sim_pearson(prefs, names[i], names[j])))
                print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!')  
        
        elif file_io == 'R' or file_io == 'r':
            print()
            file_dir = 'data/'
            datafile = 'critics_ratings.data'
            itemfile = 'critics_movies.item'
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            print('Number of users: %d\nList of users:' % len(prefs), 
                  list(prefs.keys()))
            
        elif file_io == 'P' or file_io == 'p':
            # print the u-i matrix
            print()
            if len(prefs) > 0:
                print ('Printing "%s" dictionary from file' % datafile)
                print ('User-item matrix contents: user, item, rating')
                for user in prefs:
                    for item in prefs[user]:
                        print(user, item, prefs[user][item])
            else:
                print ('Empty dictionary, R(ead) in some data!')
                
        elif file_io == 'V' or file_io == 'v':      
            print()
            if len(prefs) > 0:
                # Validate the dictionary contents ..
                print ('Validating "%s" dictionary from file' % datafile)
                print ("critics['Lisa']['Lady in the Water'] =", 
                       prefs['Lisa']['Lady in the Water']) # ==> 2.5
                print ("critics['Toby']:", prefs['Toby']) 
                # ==> {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 
                #      'Superman Returns': 4.0}
            else:
                print ('Empty dictionary, R(ead) in some data!')
                
        elif file_io == 'S' or file_io == 's':
            print()
            filename = 'critics_ratings.data'
            if len(prefs) > 0:
                data_stats(prefs, filename)
                popular_items(prefs, filename)
            else: # Make sure there is data  to process ..
                print ('Empty dictionary, R(ead) in some data!')   
        
        elif file_io == 'D' or file_io == 'd':
            print()
            if len(prefs) > 0:  
                '''
                print('Examples:')
                print ('Distance sim Lisa & Gene:', sim_distance(prefs, 'Lisa', 'Gene')) # 0.29429805508554946
                num=1
                den=(1+ sqrt( (2.5-3.0)**2 + (3.5-3.5)**2 + (3.0-1.5)**2 + (3.5-5.0)**2 + (3.0-3.0)**2 + (2.5-3.5)**2))
                print('Distance sim Lisa & Gene (check):', num/den)    
                print ('Distance sim Lisa & Michael:', sim_distance(prefs, 'Lisa', 'Michael')) # 0.4721359549995794
                print()
                '''
                print('Euclidian distance similarities:')
                
                ## add code here to calc/print User-User distance similarities 
                ## for all users or add a new function to do this
                names = list(prefs.keys())
                for i in range (len(names)):
                    for j in range (i+1, len(names)):
                        if i != j:
                            print ('Distance sim '+names[i]+' and '+names[j]+' '+str(sim_distance(prefs, names[i], names[j])))
                print()
                print('Pearson:')            
                names = list(prefs.keys())
                for i in range (len(names)):
                    for j in range (i+1, len(names)):
                        if i != j:
                            print ('Distance sim '+names[i]+' and '+names[j]+' '+str(sim_pearson(prefs, names[i], names[j])))
                
        elif file_io == 'U' or file_io == 'u':
            print()
            if len(prefs) > 0:   
                '''
                print ('Example:')
                user_name = 'Toby'
                print ('User-based CF recs for %s, sim_pearson: ' % (user_name), 
                       getRecommendations(prefs, user_name)) 
                        # [(3.3477895267131017, 'The Night Listener'), 
                        #  (2.8325499182641614, 'Lady in the Water'), 
                        #  (2.530980703765565, 'Just My Luck')]
                print ('User-based CF recs for %s, sim_distance: ' % (user_name),
                       getRecommendations(prefs, user_name, similarity=sim_distance)) 
                        # [(3.457128694491423, 'The Night Listener'), 
                        #  (2.778584003814924, 'Lady in the Water'), 
                        #  (2.422482042361917, 'Just My Luck')]
                print()
                '''
                # Calc User-based CF recommendations for all users
        
                ## add some code here to calc User-based CF recommendations 
                ## write a new function to do this ..
                def get_all_UU_recs(thisPrefs, simType, num_users, num_out):
                ##    ''' 
                ##    Print user-based CF recommendations for all users in dataset
                ##
                ##    Parameters
                ##    -- prefs: nested dictionary containing a U-I matrix
                ##    -- sim: similarity function to use (default = sim_pearson)
                ##    -- num_users: max number of users to print (default = 10)
                ##    -- top_N: max number of recommendations to print per user (default = 5)
                ##
                ##    Returns: None
                ##    '''
                    
                    names = list(thisPrefs.keys())
                    count = 0
                    for i in names :
                        print (i +" " +str(getRecommendations(thisPrefs, i, similarity=simType)))
                        count += 1
                        if count >= num_users:
                            break
                    
                
                #Pearson
                print ('User-based CF recs for sim_pearson:')
                get_all_UU_recs(prefs, sim_pearson, 10, 5)
                print()
                print ('User-based CF recs for sim_distance:')
                get_all_UU_recs(prefs, sim_distance, 10, 5)
                
                
                
                print()    
                
        elif file_io == 'LCV' or file_io == 'lcv':
            print()
            if len(prefs) > 0:             
                print ('Example:')            
                ## add some code here to calc LOOCV 
                ## write a new function to do this ..
                print('LCV w/ Pearson')
                error, error_list = loo_cv(prefs, mean_squared_error, sim_pearson, None)
                print()
                print('LCV w/ Euclidian')
                error, error_list = loo_cv(prefs, mean_squared_error, sim_distance, None)
            else:
                print ('Empty dictionary, R(ead) in some data!')   
                
        elif file_io == 'Sim' or file_io == 'sim':
            print()
            if len(prefs) > 0: 
                ready = False # sub command in progress
                sub_cmd = input('RD(ead) distance or RP(ead) pearson or WD(rite) distance or WP(rite) pearson? ')
                try:
                    if sub_cmd == 'RD' or sub_cmd == 'rd':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(open( "save_itemsim_distance.p", "rb" ))
                        sim_method = 'sim_distance'
    
                    elif sub_cmd == 'RP' or sub_cmd == 'rp':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(open( "save_itemsim_pearson.p", "rb" ))  
                        sim_method = 'sim_pearson'
                        
                    elif sub_cmd == 'WD' or sub_cmd == 'wd':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(prefs,similarity=sim_distance)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open( "save_itemsim_distance.p", "wb" ))
                        sim_method = 'sim_distance'
                        
                    elif sub_cmd == 'WP' or sub_cmd == 'wp':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(prefs,similarity=sim_pearson)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open( "save_itemsim_pearson.p", "wb" )) 
                        sim_method = 'sim_pearson'
                    
                    else:
                        print("Sim sub-command %s is invalid, try again" % sub_cmd)
                        continue
                    
                    ready = True # sub command completed successfully
                    
                except Exception as ex:
                    print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!'
                           ' Enter Sim(ilarity matrix) again and choose a Write command')
                    print()
                

                if len(itemsim) > 0 and ready == True: 
                    # Only want to print if sub command completed successfully
                    print ('Similarity matrix based on %s, len = %d' 
                           % (sim_method, len(itemsim)))
                    print()
                    ##
                    ## enter new code here, or call a new function, 
                    ##    to print the sim matrix
                    ##
                    df = pd.DataFrame(columns=list(itemsim.keys()), index=list(itemsim.keys()))
                    for i in list(itemsim.keys()):
                        for j in itemsim[i]:
                            df[i][j[1]]=j[0]
                    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                        print(df)
                    
                print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!') 
        elif file_io == 'I' or file_io == 'i':
            print()
            try:
                if len(prefs) > 0 and len(itemsim) > 0:       
                    '''
                    print ('Example:')
                    user_name = 'Toby'
        
                    print ('Item-based CF recs for %s, %s: ' % (user_name, sim_method), 
                           getRecommendedItems(prefs, itemsim, user_name)) 
                    
                    ##
                    ## Example:
                    ## Item-based CF recs for Toby, sim_distance:  
                    ##     [(3.1667425234070894, 'The Night Listener'), 
                    ##      (2.9366294028444346, 'Just My Luck'), 
                    ##      (2.868767392626467, 'Lady in the Water')]
                    ##
                    ## Example:
                    ## Item-based CF recs for Toby, sim_pearson:  
                    ##     [(3.610031066802183, 'Lady in the Water')]
                    ##
                    '''
                    print()
                    
                    print('Item-based CF recommendations for all users, '+sim_method+':')
                    # Calc Item-based CF recommendations for all users
            
                    ## add some code above main() to calc Item-based CF recommendations 
                    ## ==> write a new function to do this, as follows
                        
                    get_all_II_recs(prefs, itemsim, sim_method) # num_users=10, and top_N=5 by default  '''
                    # Note that the item_sim dictionry and the sim_method string are
                    #   setup in the main() Sim command
                    
                    ## Expected Results ..
                    
                    ## Item-based CF recs for all users, sim_distance:  
                    ## Item-based CF recommendations for all users:
                    ## Item-based CF recs for Lisa, sim_distance:  []
                    ## Item-based CF recs for Gene, sim_distance:  []
                    ## Item-based CF recs for Michael, sim_distance:  [(3.2059731906295044, 'Just My Luck'), (3.1471787551061103, 'You, Me and Dupree')]
                    ## Item-based CF recs for Claudia, sim_distance:  [(3.43454674373048, 'Lady in the Water')]
                    ## Item-based CF recs for Mick, sim_distance:  []
                    ## Item-based CF recs for Jack, sim_distance:  [(3.5810970647618663, 'Just My Luck')]
                    ## Item-based CF recs for Toby, sim_distance:  [(3.1667425234070894, 'The Night Listener'), (2.9366294028444346, 'Just My Luck'), (2.868767392626467, 'Lady in the Water')]
                    ##
                    ## Item-based CF recommendations for all users:
                    ## Item-based CF recs for Lisa, sim_pearson:  []
                    ## Item-based CF recs for Gene, sim_pearson:  []
                    ## Item-based CF recs for Michael, sim_pearson:  [(4.0, 'Just My Luck'), (3.1637361366111816, 'You, Me and Dupree')]
                    ## Item-based CF recs for Claudia, sim_pearson:  [(3.4436241497684494, 'Lady in the Water')]
                    ## Item-based CF recs for Mick, sim_pearson:  []
                    ## Item-based CF recs for Jack, sim_pearson:  [(3.0, 'Just My Luck')]
                    ## Item-based CF recs for Toby, sim_pearson:  [(3.610031066802183, 'Lady in the Water')]
                        
                    print()
                    
                else:
                    if len(prefs) == 0:
                        print ('Empty dictionary, R(ead) in some data!')
                    else:
                        print ('Empty similarity matrix, use Sim(ilarity) to create a sim matrix!')    
            except Exception as ex:
                print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!'
                           ' Enter Sim(ilarity matrix) again and choose a Write command')
                print()
                  
        elif file_io == 'RML' or file_io == 'rml':
            print()
            file_dir = 'data/ml-100k/' # path from current directory
            datafile = 'u.data' # ratngs file
            itemfile = 'u.item' # movie titles file
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            print('Number of users: %d\nList of users [0:10]:'
            % len(prefs), list(prefs.keys())[0:10] )
            
        elif file_io == 'LCVSIM' or file_io == 'lcvsim':
            print()
            sub_cmd = input('U(ser) or I(tem) based?')
            try:
                
                #NEW
                
                if sub_cmd == 'U' or sub_cmd == 'u':
                    algo = getRecommendationsSim
                elif sub_cmd == 'I' or sub_cmd == 'i':
                    algo = getRecommendedItems 
                    
                else: 
                    print ('Incorrect Command')
                 
                    
                    
                if len(prefs) > 0 and itemsim !={}:             
                    print('LOO_CV_SIM Evaluation')
                    
                    #change?
                    if len(prefs) == 7:
                        prefs_name = 'critics'
                    else:
                        prefs_name = 'not critics'
                    '''
                    metric = input ('Enter error metric: MSE, MAE, RMSE: ')
                    if metric == 'MSE' or metric == 'MAE' or metric == 'RMSE' or \
    		        metric == 'mse' or metric == 'mae' or metric == 'rmse':
                        metric = metric.upper()
                    else:
                        metric = 'MSE'
                    '''
                    #remove
                    #algo = getRecommendedItems ## Item-based recommendation
                    
                    if sim_method == 'sim_pearson': 
                        sim = sim_pearson
                        error_list  = loo_cv_sim(prefs,  sim, algo, itemsim)
                        print('%s , len(SE list): %d, using %s' 
    			  % (prefs_name,len(error_list), sim) )
                        print()
                    elif sim_method == 'sim_distance':
                        sim = sim_distance
                        error_total, error_list  = loo_cv_sim(prefs, sim, algo, itemsim)
                        print('%s:, len(SE list): %d, using %s' 
    			  % ( prefs_name,  len(error_list), sim) )
                        print()
                    else:
                        print('Run Sim(ilarity matrix) command to create/load Sim matrix!')
                    if prefs_name == 'critics':
                        print(error_list)
                else:
                    print ('Empty dictionary, run R(ead) OR Empty Sim Matrix, run Sim!')
                    
            except Exception as ex:
                print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!'
                           ' Enter Sim(ilarity matrix) again and choose a Write command')
                print()
        elif file_io == 'SIMU' or file_io == 'simu':
            print()
            if len(prefs) > 0: 
                ready = False # sub command in progress
                sub_cmd = input('RD(ead) distance or RP(ead) pearson or WD(rite) distance or WP(rite) pearson? ')
                try:
                    if sub_cmd == 'RD' or sub_cmd == 'rd':
                        # Load the dictionary back from the pickle file.
                        usersim = pickle.load(open( "save_usersim_distance.p", "rb" ))
                        sim_method = 'sim_distance'
    
                    elif sub_cmd == 'RP' or sub_cmd == 'rp':
                        # Load the dictionary back from the pickle file.
                        usersim = pickle.load(open( "save_usersim_pearson.p", "rb" ))  
                        sim_method = 'sim_pearson'
                        
                    elif sub_cmd == 'WD' or sub_cmd == 'wd':
                        # transpose the U-I matrix and calc user-user similarities matrix
                        usersim = calculateSimilarUsers(prefs,similarity=sim_distance)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open( "save_usersim_distance.p", "wb" ))
                        sim_method = 'sim_distance'
                        
                    elif sub_cmd == 'WP' or sub_cmd == 'wp':
                        # transpose the U-I matrix and calc user-user similarities matrix
                        usersim = calculateSimilarUsers(prefs,similarity=sim_pearson)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open( "save_usersim_pearson.p", "wb" )) 
                        sim_method = 'sim_pearson'
                    
                    else:
                        print("Sim sub-command %s is invalid, try again" % sub_cmd)
                        continue
                    
                    ready = True # sub command completed successfully
                    
                except Exception as ex:
                    print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!'
                           ' Enter SIMU(ilarity matrix) again and choose a Write command')
                    print()
                

                if len(usersim) > 0 and ready == True: 
                    # Only want to print if sub command completed successfully
                    print ('Similarity matrix based on %s, len = %d' 
                           % (sim_method, len(usersim)))
                    print()
                    ##
                    ## enter new code here, or call a new function, 
                    ##    to print the sim matrix
                    ##
                    ## print(itemsim)

                    for key in usersim.keys():
                        print(key)
                        i = 0
                        for value in usersim[key]:
                            print(usersim[key][i])
                            i += 1
                    print()


                print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!')
        elif file_io == 'RML' or file_io == 'rml':
            print()
            file_dir = 'data/ml-100k/' # path from current directory
            datafile = 'u.data'  # ratngs file
            itemfile = 'u.item'  # movie titles file            
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            print('Number of users: %d\nList of users [0:10]:' 
                      % len(prefs), list(prefs.keys())[0:10] )        
        else:
            done = True
    
    print('\nGoodbye!')
        
if __name__ == '__main__':
    main()