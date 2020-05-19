import numpy as np
import matplotlib.pyplot as plt



class JB:
    def __init__(self, training_set):
        # Training set is a list of numpy array with shape[ FeatureNum, FeatureDim ]
        print('Initialize Joint Bayesian Paramaters...')
        self.training_set = training_set
        self.class_num = len(self.training_set)
        self.feat_num = 0
        self.m = []
        for i in range(self.class_num):
            self.m.append(training_set[i].shape[0])
            self.feat_num += training_set[i].shape[0]

        self.feat_dim = training_set[0].shape[1]
        

        # Init the paramaters with inter and intra class cov

        mu = []
        Sum = np.zeros(self.feat_dim)
        num = []
        Sw = np.zeros((self.feat_dim,self.feat_dim))
        i=0
        for ele in training_set:
            mu_i = np.sum(ele,axis=0)
            Sum = Sum + mu_i
            mu_i = mu_i/ele.shape[0]
            mu.append(mu_i)
            num.append(ele.shape[0])
            
            for vec in ele:
                Sw = Sw + np.dot(np.reshape(vec-mu_i,(self.feat_dim,1)), np.reshape(vec-mu_i,(1,self.feat_dim)))
        Sw = Sw/self.feat_num
        
        Sum = Sum/self.feat_num
        Sb = np.zeros((self.feat_dim,self.feat_dim))
        
        i=0
        for ele in mu:
            Sb = Sb + (num[i]/self.feat_num)*np.dot(np.reshape(ele-Sum,(self.feat_dim,1)),np.reshape(ele-Sum,(1,self.feat_dim)))
            i+=1


        # Init with random positive

        '''
        tmpb = np.random.random((self.feat_dim,1))
        tmpw = np.random.random((self.feat_dim,1))
        Sb = np.dot(tmpb,tmpb.T)
        Sw = np.dot(tmpw, tmpw.T)
        '''

        #Init with eye(n)
        '''
        Sb=np.eye(self.feat_dim)
        Sw=np.eye(self.feat_dim)
        '''
        self.S_mu = Sb
        self.S_ep = Sw
        
    #one iter for EM train procedure
    def EM_OneStep(self):
        S_ep = np.zeros([self.feat_dim, self.feat_dim])
        S_mu = np.zeros([self.feat_dim, self.feat_dim])

        F = np.linalg.pinv(self.S_ep)
        G = []
        for i in range(self.class_num):
            G.append( np.dot(np.dot(-np.linalg.pinv(self.m[i]*self.S_mu+self.S_ep),self.S_mu),F) )
        pos = 0
        tmp3 = np.dot(self.S_ep, F)
        for i in range(self.class_num):
            #print(i)
            SmuFG = np.dot(self.S_mu,(F+self.m[i]*G[i]))
            SepG = np.dot(self.S_ep, G[i])
            
            tmp1 = (np.dot(SmuFG, np.sum(self.training_set[i],axis=0))).reshape(self.feat_dim,1)
            S_mu = S_mu+np.dot(tmp1,tmp1.T)

            tmp2 = np.dot(SepG, np.sum(self.training_set[i],axis=0))
            for j in range(self.m[i]):
                tmp = (np.dot(tmp3, self.training_set[i][j]) + tmp2).reshape(self.feat_dim,1)
                S_ep = S_ep + np.dot(tmp,tmp.T)
        
        self.S_mu = S_mu/self.class_num
        self.S_ep = S_ep/self.feat_num
        
    def train(self, iter_num):
        # Train the model
        convergence_mu=[]
        convergence_ep=[]
        print('Strat EM training, this procedure sould take some time...')
        for i in range(iter_num):
            old_S_mu = self.S_mu
            old_S_ep = self.S_ep
            
            self.EM_OneStep()
            
            convergence_mu.append( np.linalg.norm(self.S_mu-old_S_mu)/np.linalg.norm(self.S_mu))
            convergence_ep.append( np.linalg.norm(self.S_ep-old_S_ep)/np.linalg.norm(self.S_ep))
            print("Iter #{0}: Change_of_Smu = {1} Change_of_Sep = {2}".format(i+1,convergence_mu[i],convergence_ep[i]))
            
        #plt.plot(convergence_mu,label = 'convergence_mu')
        #plt.legend()
        #plt.xlabel('Iteration Number')
        #plt.ylabel('Change of S_mu')
        #plt.show()
        #plt.plot(convergence_ep,label = 'convergence_ep')
        #plt.legend()
        #plt.xlabel('Iteration Number')
        #plt.ylabel('Change of S_ep')
        #plt.show()
        
        # F = np.linalg.pinv(self.S_ep)
        # G = np.dot(np.dot(-np.linalg.pinv(2*self.S_mu+self.S_ep),self.S_mu),F)
        # self.A = np.linalg.pinv(self.S_mu+self.S_ep) - (F + G)
        # self.G = G
        
    def Store(self, Path_mu, Path_ep):
        np.save(Path_mu,self.S_mu)
        print(Path_mu)
        np.save(Path_ep,self.S_ep)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
