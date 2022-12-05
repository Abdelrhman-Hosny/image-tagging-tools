import cvxpy as cp

class recognizer():

    def __init__(self, dataset):
        self.data = dataset
        self.train = dataset.train[:,1:].T


    def solve(self, b, y, epsilon):


        self.x = cp.Variable(b.shape[0])
        self.obj = cp.Minimize(cp.norm(self.x,1))
        self.constraints = [cp.norm(self.train*self.x - y,2) <= epsilon]
        self.prob = cp.Problem(self.obj, self.constraints)
        try:
            self.prob.solve()
        except cp.error.SolverError:
            self.prob.solve(solver="SCS")
    def getOptim(self):
        return self.x.value
    def getOptimVal(self):
        return self.prob.value
    def getStatus(self):
        return self.prob.status

