import threading

### Model study classes ###

class BaselineStudy(threading.Thread):
    def __init__(self, data):
        threading.Thread.__init__(self)
        self.data = data
        print "study initialized"

    def fit(self):
        return

    def pred(self):
        print "making predictions for baselines"

    def eval(self):
        print "evaluating predicitions"

    def run(self):
        self.fit()
        self.pred()
        self.eval()


class MFStudy(BaselineStudy):
    def fit(self):
        print "fitting a MF model with datat(%s)" % self.data

    def pred(self):
        print "making predicitions for MF model"


if __name__ == '__main__':
    #TODO: process input somehow, create needed output dirs
    data = "data object"

    # fit, predict, and evaluate for each model
    baselines = BaselineStudy(data) 
    baselines.start()

    mf = MFStudy(data)
    mf.start()

    # wait for each one to finish
    baselines.join()
    mf.join()

    print "aggregate results here"
