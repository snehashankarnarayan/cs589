#!/usr/bin/python

__author__ = 'snehas'

import sys
import kmeans
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

data_dir = ""

#Optimal model
global_k = 3
global_rseed = 4

#Specify data directories
MACH = '/Users/snehas/vagroot/shared_files/data/HW4/'
EVAL = '/vagrant/shared_files/data/HW4/'

#Some globals
Xtest = []
Xtrain = []
Xval = []
Users = []
Items = []
Genres = []

def load_data():
    global Xval, Xtrain, Xtest, Users, Items, Genres

    #Load the training ratings
    A       = np.load(data_dir + "train.npy")
    A.shape = (1,)
    Xtrain  = A[0]

    #Load the validation ratings
    A       = np.load(data_dir + "validate.npy")
    A.shape = (1,)
    Xval    = A[0]

    #Load the test ratings
    A       = np.load(data_dir + "test.npy")
    A.shape = (1,)
    Xtest   = A[0]

    #Load the user, item, and genre information
    Users   = np.load(data_dir + "users.npy")
    Items   = np.load(data_dir + "items.npy")
    Genres  = np.load(data_dir + "genres.npy")

#Used for Question 1
def run_clustering(k, r):
    #Train k-Means on the training data
    model = kmeans.kmeans(n_clusters=k, random_seed=1000*r + 7, verbose=False)
    model.fit(Xtrain)

    #Predict back the training ratings and compute the RMSE
    XtrainHat = model.predict(Xtrain, Xtrain)
    tr= model.rmse(Xtrain, XtrainHat)

    #Predict the validation ratings and compute the RMSE
    XvalHat = model.predict(Xtrain, Xval)
    val= model.rmse(Xval, XvalHat)

    return tr, val

#Run kmeans for Train and Validation sets over a range of values for k and a range of random seeds
def q1a():
    k_list = [2,3,4,5,6, 7, 11, 18, 25, 29, 50, 65, 80, 90, 100]
    for k in k_list:
        tr_list = []
        val_list = []
        #8 random restarts
        for r in range(0, 8):
            tr, val = run_clustering(k, r)
            tr_list.append(tr)
            val_list.append(val)
        print("%d %.7f %.7f %.7f %.7f %.7f %.7f")%(k, max(tr_list), min(tr_list), np.mean(tr_list), max(val_list), min(val_list), np.mean(val_list))

#Run kmeans over select values of k to figure out what's the optimal k
def q1b():
    k_list = [2,3,4]
    min_k = 0
    min_r = 0
    min_rmse = 9999
    val_list = dict()
    for k in k_list:
        #8 random restarts
        for r in range(0, 8):
            tr, val = run_clustering(k, r)
            if(val < min_rmse):
                min_k = k
                min_r = r
                min_rmse = val
            val_list[k,r] = val
    print val_list
    print min_k, min_r, min_rmse

#Compute RMSE using optimal model
def q1b_rmse():
    model = get_model()
    #Predict the validation ratings and compute the RMSE
    XtestHat = model.predict(Xtrain, Xtest)
    te = model.rmse(Xtest, XtestHat)
    print te


#Get the zipcode labels per state
def get_zip_labels(l):
    zip_states = ["" for x in range(len(l))]
    zip_states[l.index('0')] = 'CT, MA, ME, NH, NJ, NY, PR, RI, VT'
    zip_states[l.index('1')] = 'DE, NY, PA '
    zip_states[l.index('2')] = 'DC, MD, NC, SC, VA, WV'
    zip_states[l.index('3')] = 'AL, FL, GA, MS, TN'
    zip_states[l.index('4')] = 'IN, KY, MI, OH'
    zip_states[l.index('5')] = 'IA, MN, MT, ND, SD, WI'
    zip_states[l.index('6')] = 'IL, KS, MO, NE'
    zip_states[l.index('7')] = 'AR, LA, OK, TX'
    zip_states[l.index('8')] = 'AZ, CO, ID, NM, NV, UT, WY'
    zip_states[l.index('9')] = 'AK, CA, HI, OR, WA'
    return zip_states

#Get zipcode info
def get_state_code(r):
    if r=='0':
        return 'CT, MA, ME, NH, NJ, NY, PR, RI, VT'
    elif r=='1':
        return 'DE, NY, PA '
    elif r=='2':
        return 'DC, MD, NC, SC, VA, WV'
    elif r=='3':
        return 'AL, FL, GA, MS, TN'
    elif r=='4':
        return 'IN, KY, MI, OH'
    elif r=='5':
        return 'IA, MN, MT, ND, SD, WI'
    elif r=='6':
        return 'IL, KS, MO, NE'
    elif r=='7':
        return 'AR, LA, OK, TX'
    elif r=='8':
        return 'AZ, CO, ID, NM, NV, UT, WY'
    elif r=='9':
        return 'AK, CA, HI, OR, WA'


#Code for question 3
def q3():
    model = get_model()
    centers = model.get_centers()

    font = {'size'   : 8}
    plt.rc('font', **font)

    movie_list = dict()
    genre_list = dict()

    for i in range(0, global_k):
        movie_list[i] = list()
        genre_list[i] = list()

    #Get the movie title and genre data
    for i in range(0, global_k):
        for j in range(0, len(centers[i])):
            tuple_item = (centers[i][j], Items[j][1])
            movie_list[i].append(tuple_item)

            tuple_item = (centers[i][j], Items[j][2])
            genre_list[i].append(tuple_item)

    for i in range(0, global_k):
        #Question 3a
        sorted_list = sorted(movie_list[i], key = lambda x:x[0],reverse=True)
        print "Highest ratings: Cluster: " + str(i) + str(sorted_list[0:5])

        #Question 3b
        sorted_list = sorted(movie_list[i], key = lambda x:x[0],reverse=False)
        print "Lowest ratings: Cluster: " + str(i) + str(sorted_list[0:5])

        #Question 3c
        rating4_list = list(elem for elem in genre_list[i] if elem[0] >= 4.0)
        genre_count_list = np.zeros(len(Genres))

        #Add up all the genres
        for elem in rating4_list:
            genre_count_list = np.add(genre_count_list, elem[1])

        #Convert to percentage
        genre_count_list = np.multiply(genre_count_list, 100.0/len(rating4_list))
        plt.close()
        fig = plt.figure()
        ax = plt.subplot(111)
        width=0.45
        ax.bar(np.arange(len(Genres)), genre_count_list, width=width, color = '#009AB2', align="center")
        ax.set_xticks(np.arange(len(Genres)) + width/2)
        ax.set_xticklabels(Genres)
        plt.title("Percentage of 4+ rated movies in each Genre Cluster ID" + str(i))
        plt.xlabel("Genre")
        plt.ylabel("Percentage")
        fig.autofmt_xdate()
        fig.savefig('q3c_' + str(i) + '.eps')

        #Question 3c
        rating2_list = list(elem for elem in genre_list[i] if elem[0] <= 2.0)
        genre_count_list = np.zeros(len(Genres))
        #Add up all genres
        for elem in rating2_list:
            genre_count_list = np.add(genre_count_list, elem[1])
        #Convert to percentage
        genre_count_list = np.multiply(genre_count_list, 100.0/len(rating2_list))
        plt.close()
        fig = plt.figure()
        ax = plt.subplot(111)
        width=0.6
        ax.bar(np.arange(len(Genres)), genre_count_list, width=width, color = '#F48533', align="center")
        ax.set_xticks(np.arange(len(Genres)) + width/2)
        ax.set_xticklabels(Genres)
        plt.title("Percentage of 2- rated movies in each Genre Cluster ID" + str(i))
        plt.xlabel("Genre")
        plt.ylabel("Percentage")
        fig.autofmt_xdate()
        fig.savefig('q3d_' + str(i) + '.eps')

#Save optimal model.npy file
def q4():
    model = get_model()
    filename = 'model.npy'
    np.save(filename, model.get_centers())

#All code for question 2
def q2():
    #Get the graph for data cases
    font = {'size'   : 8}
    plt.rc('font', **font)
    model = get_model()
    z = model.cluster(Xtest)
    freq = Counter(z)

    #Get points for each cluster
    data = [freq[0],freq[1],freq[2]]
    fig = plt.figure()
    ax = plt.subplot(111)
    width=0.6
    clusters = range(0, global_k, 1)
    #Put it in a graph
    ax.bar(np.arange(len(clusters)), data, width=width, color = '#C44441', align="center")
    ax.set_xticks(np.arange(len(clusters)) + width/2)
    ax.set_xticklabels(clusters)
    plt.title("Number of data cases in each cluster")
    plt.xlabel("Cluster IDs")
    plt.ylabel("No of data cases")
    fig.savefig('q2a.eps')

    #Initialize for Question 2b - 2e
    age_list = dict()
    gender_list = dict()
    work_list = dict()
    zip_list = dict()
    for i in range(0, global_k):
        age_list[i] = list()
        gender_list[i] = list()
        work_list[i] = list()
        zip_list[i] = list()


    for i in range(0, len(z)):
        cur_list = age_list[int(z[i])]
        #Age is in index 1
        cur_list.append(Users[i][1])
        age_list[int(z[i])] = cur_list

        cur_list = gender_list[int(z[i])]
        #Gender is in index 2
        cur_list.append(Users[i][2])
        gender_list[int(z[i])] = cur_list

        cur_list = work_list[int(z[i])]
        #Work is in index 3
        cur_list.append(Users[i][3])
        work_list[int(z[i])] = cur_list

        cur_list = zip_list[int(z[i])]
        #Work is in index 3
        cur_list.append(Users[i][4])
        zip_list[int(z[i])] = cur_list

    #Do for each K
    for i in range(0, global_k):
        #Question 2b
        plt.close()
        fig.clf()
        data = age_list[i]
        fig = plt.figure()
        plt.hist(data, bins=range(0,110,10), color='#7A5892', histtype='bar')
        plt.xticks(range(0,110,10))
        plt.title("Age of users in Cluster ID" + str(i))
        plt.xlabel("Age")
        plt.ylabel("No of users")
        fig.savefig('q2b_' + str(i) + '.eps')

        #Question 2c
        plt.close()
        fig.clf()
        #Get gender data
        freq = Counter(gender_list[i])
        data = [freq[0],freq[1]]
        fig = plt.figure()
        ax = plt.subplot(111)
        width=0.6
        genders = ['Female','Male']
        ax.bar(np.arange(len(genders)), data, width=width, color = '#7A5892', align="center")
        ax.set_xticks(np.arange(len(genders)) + width/2)
        ax.set_xticklabels(genders)
        plt.title("Gender of users in Cluster ID" + str(i))
        plt.xlabel("Gender")
        plt.ylabel("No of users")
        fig.savefig('q2c_' + str(i) + '.eps')

        #Question 2d
        plt.close()
        fig.clf()
        freq = Counter(work_list[i])
        #Do sorting for better charts
        freq = freq.most_common()
        freq_labels = [r[0] for r in freq]
        freq_values = [r[1] for r in freq]
        fig = plt.figure()
        color=plt.cm.rainbow(np.linspace(0, 1, len(freq_labels)))
        plt.pie(freq_values, colors=color, startangle=90)
        plt.axis('equal')
        plt.legend(freq_labels)
        plt.title("Occupation of users in Cluster ID" + str(i))
        fig.savefig('q2d_' + str(i) + '.eps')

        #Question 2e
        plt.close()
        fig.clf()
        font = {'size'   : 8}
        plt.rc('font', **font)
        aggr_ziplist = list()
        #Extract first digits
        for val in zip_list[i]:
            if(val[:1].isdigit()):
                aggr_ziplist.append(val[:1])
        freq = Counter(aggr_ziplist)
        freq = freq.most_common()
        #Get the appropriate labels for the states
        freq_labels = [get_state_code(r[0]) for r in freq]
        freq_values = [r[1] for r in freq]
        #freq_labels = get_zip_labels(freq.keys())
        #freq_values = freq.values()
        fig = plt.figure()
        color=plt.cm.rainbow(np.linspace(0, 1, len(freq_labels)))
        plt.pie(freq_values, colors=color, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.legend(freq_labels, loc = 'best')
        plt.title("Zipcodes of users in Cluster ID" + str(i))
        fig.savefig('q2e_' + str(i) + '.eps')

#Main module for running code per question
def cluster():
    print "Question 1a"
    q1a()
    print "Question 1b"
    q1b()
    print 'RMSE on test data'
    q1b_rmse()
    print "Question 2"
    q2()
    print "Question 3"
    q3()
    print "Question 4"
    q4()

#Get model that is chosen
def get_model():
    r = global_rseed
    k = global_k
    model = kmeans.kmeans(n_clusters=k, random_seed=1000*r + 7)
    model.fit(Xtrain)
    return model

def run_me(loc):
    global data_dir
    if(loc == "mach"):
        data_dir = MACH
        load_data()
        cluster()
    elif(loc == "eval"):
        data_dir = EVAL
        load_data()
        cluster()

if __name__ == "__main__":
    loc = sys.argv[1]
    global data_dir
    if(loc == "mach"):
        data_dir = MACH
        load_data()
        cluster()
    elif(loc == "eval"):
        data_dir = EVAL
        load_data()
        cluster()