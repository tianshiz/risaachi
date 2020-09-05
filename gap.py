class DetK(KPlusPlus):
    def fK(self, thisk, Skm1=0):
        X = self.X
        Nd = len(X[0])
        a = lambda k, Nd: 1 - 3/(4*Nd) if k == 2 else a(k-1, Nd) + (1-a(k-1, Nd))/6
        self.find_centers(thisk, method='++')
        mu, clusters = self.mu, self.clusters
        Sk = sum([np.linalg.norm(mu[i]-c)**2 \
                 for i in range(thisk) for c in clusters[i]])
        if thisk == 1:
            fs = 1
        elif Skm1 == 0:
            fs = 1
        else:
            fs = Sk/(a(thisk,Nd)*Skm1)
        return fs, Sk   