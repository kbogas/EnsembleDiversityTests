class DiversityTests(object):

    """
    Class Wrapper to get Diversity Measures over collection of predictions.
    Args:
        @predictions: list of lists. Each sublist contains the predictions
                      of a classifier
        @names: list of strings. Each string is the name of the classifier.
        @true: list of labels. Each label is the truth label
    """

    def __init__(self, predictions, names, true):

        N = len(true)
        labels = set(true)
        if len(predictions) != len(names):
            raise AttributeError('Number of classifiers is different than number \
                                  of names. %d != %d.' % (len(predictions), len(names)))
        for i, predict in enumerate(predictions):
            if len(predict) != N:
                raise AttributeError('Number of predictions of classifier %s is different\
                                      then the number of true labels. %d != %d .\
                                      ' % (names[i], len(predict), N))
            if labels.isdisjoint(set(predict)):
                raise AttributeError('Label in predictions of %s not in truth set. \
                                      ' % (names[i]))
        self.predictions = predictions
        self.names = names
        self.true = true

    def print_measures_details(self):

        """Print details about the computed measures"""

        print '==============================================================='
        print 'Correlation: For +-1 perfect aggrement/disagreement'
        print 'Q-statistic: Q=0  => Independent. For q>0 predictors find the the same results'
        print "Cohen's k: k->0  => High Disagreement => High Diversity"
        print "Kohovi-Wolpert Variance -> Inf => High Diversity"
        print "Conditional Accuracy Table: Conditional Probability that the row system predicts correctly, given"
        print "                            that the column system also predicts correctly"
        print '==============================================================='

        return

    def get_KWVariance(self, print_flag=True):

        """Get Kohovi Wolpert Variance."""

        kw = KW_Variance(self.predictions, self.true)
        if print_flag:
            print '#####  Kohovi-Wolpert Variance: % 0.3f  #####' % kw
        return kw

    def get_avg_pairwise(self, print_flag=True):

        """Get average over all pairs in collection for different measures."""

        avg_metrics = avg_pairwise(self.predictions, self.names, self.true)
        if print_flag:
            print '#### Pairwise Average Metrics: #####'
            print 'Avg. Cor: %0.3f' % avg_metrics[0]
            print 'Avg. Q-statistic: %0.3f' % avg_metrics[1]
            print "Avg. Cohen's k: %0.3f" % avg_metrics[2]
        return avg_metrics

    def get_conditional_acc_table(self, print_flag=True):

        """Get the Conditional Accuracy Matric."""

        df_table = acc_cont_table(self.predictions, self.names, self.true, False)
        if print_flag:
            print '###Conditional Accuracy Table###'
            print df_table.astype('float').to_string(float_format=lambda x: '%0.2f' % (x))
        return df_table

    def print_report(self):

        print '---------------------------------------------------------------'
        print 'Diversity Tests Report'
        print '---------------------------------------------------------------\n'
        print 'Measures Details'
        self.print_measures_details()
        print '---------------------------------------------------------------\n'
        print 'Measures Results'
        print '---------------------------------------------------------------\n'
        
        methods = [method for method in dir(self) if callable(getattr(self, method)) and 'get_' in method]
        print methods
        for method in methods:
            method_run = getattr(self, method)
            method_run()
            print '- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \n'
        return  
        
    
    def help(self):
        """Just a helper function to print the class docstring."""
        
        return self.__doc__

def Pairwise_Tests(y_a, y_b, y_true, name_1, name_2, p=0.05, print_cont = False):
    
    """ Method for calculating diversity measures of pairs of classifiers. 
        Also prints McNemar Contigency Table(if wanted) and calculates if
        there is statistical significant difference in results.
    
       input:
            @ y_a : list of predictions for classifiear a
            @ y_b : list of predictions for classifiear b
            @ y_true: list of ground trutrh labels
            @ name_1: name of classifier a
            @ name_2: name of classifier b
            @ p: 1-p=is confidence value for chi2 test
            @ print_cont: print McNemar table for oracle outputs between a,b
       
       output:
            @ correlation: Phi coefficient
            @ q_statistic: Q-test coeff
            @ cohens_k: Interater agreement
            @ a_given_b: Conditional Probability of classifier a predicting 
                         correctly the correctly predicted, by b, instances
            @ b_given_a: Conditional Probability of classifier b predicting 
                         correctly the correctly predicted, by a, instances
    """
    
    from numpy import zeros
    from pandas import DataFrame
    from scipy.stats import chi2
    from math import sqrt
    
    
    
    tresh = chi2.ppf(q = 1-p, # Find the critical value for 95% confidence*
                      df = 1)   # Df = number of variable categories - 1
    
    table_scores = numpy.zeros([2,2], dtype=float)  # 2x2 McNemart Oracle Table
    counts_a = 0 # correct classifier a predictions
    counts_b = 0 # correct classifier b predictions
    for i, y in enumerate(y_true):
        if y == y_a[i]:
            counts_a +=1
            if y == y_b[i]:
                counts_b +=1
                table_scores[0,0] += 1  # both a and b correct
            else:
                table_scores[0,1] += 1 # a correct, b wrong
        else:
            if y == y_b[i]:
                counts_b +=1
                table_scores[1,0] += 1 # b wrong, a correct
            else:  
                table_scores[1,1] += 1 # a, b both wrong
    b_given_a = table_scores[0,0]/float(counts_a)
    a_given_b = table_scores[0,0]/float(counts_b)
    
    if table_scores[0,1]+table_scores[1,0] == 0: # can't find chi2
        chi_squared_value = 0
    else: # calcualate chi2 value if possible
        chi_squared_value = pow(abs(table_scores[0,1]-table_scores[1,0])-1,2)/float(table_scores[0,1]+table_scores[1,0])
    
    # Correlation between predictions
    correlation = (table_scores[0,0]*table_scores[1,1]-table_scores[0,1]*table_scores[1,0])/float(sqrt((table_scores[0,0]*table_scores[0,1])*(table_scores[1,0]*table_scores[1,1])*(table_scores[0,0]*table_scores[1,0])*(table_scores[0,1]*table_scores[1,1]) ))
    
    # Q statistic
    q_statistic = (table_scores[0,0]*table_scores[1,1]-table_scores[0,1]*table_scores[1,0])/float(table_scores[0,0]*table_scores[1,1]+table_scores[0,1]*table_scores[1,0])
    
    # Interater Agreement of predictions
    cohens_k = (table_scores[0,0]*table_scores[1,0]-table_scores[0,1]*table_scores[1,1])/float((table_scores[0,0]*table_scores[0,1])*(table_scores[1,0]+table_scores[1,1])+(table_scores[0,0]+table_scores[1,0])*(table_scores[0,1] +table_scores[1,1]))

    # Turn absolute counts in percentages for McNemar Table printing
    table_scores /= float(len(y_true))
    table_scores *= 100
    
    # Check for statistical independece according to chi2 test
    if chi_squared_value > tresh: 
        '!!! p=%0.2f Significant Difference: chi2-value %0.3f > %0.3f !!!' %(p, chi_squared_value, tresh)
    if print_cont: # print McNemar Table test
        print 'Contigency Table: %s -%s' %(name_1, name_2)
        print '====================='
        df = DataFrame(table_scores, [name_1+'-cor', name_1+'-wro'], [name_2+'-cor', name_2+'-wro'])
        print df.astype('float').to_string(float_format= lambda x: '%0.2f'%(x))
        print '====================='
    return correlation, q_statistic, cohens_k, a_given_b, b_given_a


def KW_Variance(predictions, y_true):
    
    """Modification by Kuncheva et al. Expects a list of lists, containing predictions and a ground truth list."""

    correct_count = [0 for i in xrange(len(y_true))] # initialize correct counts per sample
    final_score = 0 # score to return
    for i, y in enumerate(y_true): # for each sample
        for pred in predictions: # check for each classifier
            if y == list(pred)[i]:
                correct_count[i] += 1 # if the prediction was correct, update counter
        final_score += correct_count[i]*(len(predictions)-correct_count[i]) # sum over all classifiers for this count

    return final_score/float(len(y_true)*pow(len(predictions),2))
                
    
def avg_pairwise(predictions, names, true):
    
    "Average of the pairwise metrics defined in PairWise Tests"
    
    avg_metrics = [0,0,0] # Differente Metrics Initialization
    num_pairs = 0 # Normalization factor
    for i in xrange(len(predictions)):
        for j in range(i+1, len(predictions)): # for each pair
            num_pairs +=1
            cor, q, k, _, _ = Pairwise_Tests(predictions[i], predictions[j], true, print_names[i], print_names[j])
            if type(cor) is float: # check to avoid when cor is infinite
                avg_metrics[0] += float(cor)
            avg_metrics[1] += q
            avg_metrics[2] += k
    avg_metrics = [avg/float(num_pairs) for avg in avg_metrics] # normalize over all pairs
    return avg_metrics    

def acc_cont_table(predictions, names, true, print_flag=True):
    
    """Create Conditional Accuracy Tables as in:
       Combining Information Extraction Systems Using Voting and Stacked Generalization
       by Sigletos et al, 2005"""
    
    from numpy import eye
    from pandas import DataFrame
    
    table = eye(len(predictions)) # table initilization
    for i in xrange(len(predictions)):
        for j in range(i+1, len(predictions)): # for each pair
            _, _, _, i_given_j, j_given_i = Pairwise_Tests(predictions[i], predictions[j], true, names[i], names[j])
            table[i, j] = i_given_j
            table[j, i] = j_given_i
    df = DataFrame(table, names, names)
    if print_flag:
        print df.astype('float').to_string(float_format= lambda x: '%0.2f'%(x))
    return df