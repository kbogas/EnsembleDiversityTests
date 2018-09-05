class BaseClassifiers(object):

    """
    Class Wrapper to get an overview of the predictions of base classifiers.
    Args:
        @predictions: list of lists. Each sublist contains the predictions
                      of a classifier
        @names: list of strings. Each string is the name of the classifier.
        @true: list of labels. Each label is the truth label
        @vis_flag: boolean flag. If True will also plot bar charts depicting
                   the information printed. Defaults to False for readability.
    """

    def __init__(self, predictions, names, true, vis_flag=False):

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
        self.vis_flag = vis_flag

    def get_comparison_report(self):

        _, fig = comparison_report(self.predictions, self.names, self.true, self.vis_flag)
        if self.vis_flag:
            pass
            #import plotly.plotly as py

            #py.iplot(fig)
        return

    def get_classification_report(self):

        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

        print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CLASSIFICATION RESULTS PER MODEL~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        for i, predict in enumerate(self.predictions):
            print "============================ Model: %s ==================================+" % str(self.names[i])
            acc = accuracy_score(self.true, predict)
            conf = confusion_matrix(self.true, predict, labels=sorted(list(set(self.true))))
            rep = classification_report(self.true, predict, target_names=sorted(list(set(self.true))))
            print('Accuracy : {}'.format(acc))
            print('Confusion matrix :\n {}'.format(conf))
            print('Classification report :\n {}'.format(rep))
            print "============================================================================"
        return

    def get_mistakes_clustering(self):

        mistakes_clustering(self.predictions, self.names, self.true)
        return

    def mistakes_clustering(predictions, names, true):

        pass

    def get_per_class_accuracy(self):
        print "Per Class Accuracy of the models"
        df = per_class_accuracy(self.predictions, self.names, self.true)
        return df
    
    def get_per_class_f1(self):
        print "Per Class Accuracy of the models"
        df = per_class_f1(self.predictions, self.names, self.true)
        return df


    def help(self):

        """Just a helper function to print the class docstring."""
        return self.__doc__



def per_class_accuracy(predictions, names, true):

    from pandas import DataFrame
    from numpy import array, hstack, zeros

    Num_models = len(predictions)
    labels = sorted(list(set(true)))
    Num_samples = len(labels)
    tmp = array(predictions[0]).reshape(-1,1)
    for i in xrange(1, len(predictions)):
        #print tmp
        #print tmp.shape
        #print array(predictions[i].reshape(-1,1))
        #print array(predictions[i].reshape(-1,1)).shape
        tmp = hstack((tmp, array(predictions[i]).reshape(-1,1)))
    tmp = hstack((tmp, array(true).reshape(-1,1)))
    df = DataFrame(tmp, columns=names + ['true'])
    counter = zeros([Num_models, len(labels)])
    for i, name in enumerate(names):
        df1 = df[df[name] == df['true']]
        for j, label in enumerate(labels):
            #print df1[df1['true'] == label].shape[0]
            #print float(len([lab_ for lab_ in labels if lab_ == label]))
            counter[i, j] = 100*df1[df1['true'] == label].shape[0] / float(len([lab_ for lab_ in true if lab_ == label]))
    final_df = DataFrame(counter, columns = labels, index = names)
    print final_df
    return final_df

def per_class_f1(predictions, names, true):

    from pandas import DataFrame
    from numpy import array, hstack, zeros, vstack
    from sklearn.metrics import f1_score

    labels = sorted(list(set(true)))
    counter = f1_score(predictions[0], true, average=None, labels=labels).reshape(1,-1)
    print counter.shape
    for i in xrange(1, len(predictions)):
        counter = vstack((counter,f1_score(predictions[i], true, average=None, labels=labels).reshape(1,-1)))
    print counter.shape
    final_df = DataFrame(counter, columns = labels, index = names)
    print final_df
    return final_df










def comparison_report(predictions, names, true, print_flag=False):

    import numpy, pandas
    import matplotlib.pyplot as plt
    
    fig1 = None
    N = len(true)
    L = len(predictions)
    num__pairs = L*(L-1)/2
    correct = numpy.zeros([L,N])
    correct_per = []
    for i in xrange(L):
        correct[i, :] =  numpy.core.defchararray.equal(predictions[i], true)
        correct_per.append(numpy.sum(correct[i,:])/float(N))
    #print correct.shape
    #print 
    df = pandas.DataFrame(correct.T, columns = names)
    #print df.head(20)
    # Classifier Performance
    print 'Base Accuracies'
    acc = []
    acc_s = ''
    for name in names:
        acc.append(df.sum(axis=0)[name]*100/float(N))
        acc_s += '%s : %0.2f  ||  ' % (name, acc[-1])
    acc_s = acc_s[:-4]
    print acc_s
    
    if print_flag:
        fig, ax = plt.subplots()
        s = plt.bar([i for i in xrange(L)], acc, align='center', alpha=0.4)
        plt.xticks([i for i in xrange(L)], names)
        plt.ylabel('% Accuracy')
        plt.title('Base Classifiers')
        autolabel(s.patches, ax)
        plt.show()

    # Classifier versus the others
    count_others = numpy.zeros([L,L])
    for i, name1 in enumerate(names):
        tmp = df[df.sum(axis=1)==1]
        if not(tmp[tmp[name1]>0].empty):
            if tmp[tmp[name1]>0].shape[0] == 0:
                count_others[0, i] = 1*100/float(N)
            else:
                count_others[0, i] = tmp[tmp[name1]>0].shape[0]*100/float(N)
    pairs_titles = [[] for i in xrange(0,L)]
    pairs_titles[0] = names
    #print pairs_titles
    for i in xrange(0, L):
        name1 = names[i]
        #print name1
        cyclic_names = [name1]+names[names.index(name1)+1:]+names[:names.index(name1)]
        tmp = df[df.sum(axis=1)==2]
        tmp2 = tmp[tmp[name1]>0]
        #tmp2 = df[df[name1]>0]
        #N_class = df[df[name1]>0].shape[0]
        N_class = tmp2.shape[0]
        #print tmp2
        #print df[df[name1]>0]
        for j in xrange(1, L):
            name2 = cyclic_names[j]
            #print name2            
            if not(tmp2[tmp2[name2]>0].empty):
                if tmp2[tmp2[name2]>0].shape[0] == 0:
                     count_others[j, i] = 1*100/float(N_class)
                else:
                    count_others[j, i] = tmp2[tmp2[name2]>0].shape[0]*100/float(N_class)
            #if name1 == 'soac':
                #print name2
                #print tmp2[tmp2[name2]>0]
            pairs_titles[j].append(name1+ '-'+name2)
        #print count_others[:,i]
    #print count_others
    count_pd = pandas.DataFrame(count_others.T, index=names, columns=['Only this Model']+ [' %d-model aggree' % i for i in xrange(1,L)])
    print 'Models Correct Aggrement Percentages'
    print count_pd.astype('float').to_string(float_format= lambda x: '%0.2f'%(x))
    if print_flag:
        import plotly.graph_objs as go
        from matplotlib.pyplot import cm
        
        top_labels = ['Only this Model']+ [' %d-model aggree' % i for i in xrange(1,L)]

        colors = list(iter(cm.rainbow(numpy.linspace(0, 1, L))))

        x_data = count_others.T.tolist()[::-1]

        #print x_data
        y_data = ['The course was effectively<br>organized',
                  'The course developed my<br>abilities and skills ' +
                  'for<br>the subject', 'The course developed ' +
                  'my<br>ability to think critically about<br>the subject',
                  'I would recommend this<br>course to a friend']

        y_data = names[::-1]
        #print y_data
        #print top_labels

        traces = []

        for i in range(0, len(x_data[0])):
            for xd, yd in zip(x_data, y_data):
                traces.append(go.Bar(
                    x=xd[i],
                    y=yd,
                    orientation='h',
                    marker=dict(
                        color=colors[i],
                        line=dict(
                                color='rgb(248, 248, 249)',
                                width=1)
                    )
                ))

        layout = go.Layout(
            xaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=False,
                zeroline=False,
                domain=[0.15, 1]
            ),
            yaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=False,
                zeroline=False,
            ),
            barmode='stack',
            paper_bgcolor='rgb(248, 248, 255)',
            plot_bgcolor='rgb(248, 248, 255)',
            margin=dict(
                l=120,
                r=10,
                t=140,
                b=80
            ),
            showlegend=False,
        )

        annotations = []

        for yd, xd in zip(y_data, x_data):
            # labeling the y-axis
            annotations.append(dict(xref='paper', yref='y',
                                    x=0.14, y=yd,
                                    xanchor='right',
                                    text=str(yd),
                                    font=dict(family='Arial', size=14,
                                              color='rgb(67, 67, 67)'),
                                    showarrow=False, align='right'))
            # labeling the first percentage of each bar (x_axis)
            annotations.append(dict(xref='x', yref='y',
                                    x=xd[0] / 2, y=yd,
                                    text='%0.2f'%xd[0],
                                    font=dict(family='Arial', size=14,
                                              color='rgb(248, 248, 255)'),
                                    showarrow=False))
            # labeling the first Likert scale (on the top)
            if yd == y_data[-1]:
                annotations.append(dict(xref='x', yref='paper',
                                        x=xd[0] / 2, y=1.1,
                                        text=top_labels[0],
                                        font=dict(family='Arial', size=14,
                                                  color='rgb(67, 67, 67)'),
                                        showarrow=False))
            space = xd[0]
            for i in range(1, len(xd)):
                    # labeling the rest of percentages for each bar (x_axis)
                    annotations.append(dict(xref='x', yref='y',
                                            x=space + (xd[i]/2), y=yd, 
                                            text='%0.2f ' % xd[i],
                                            font=dict(family='Arial', size=14,
                                                      color='rgb(248, 248, 255)'),
                                            showarrow=False))
                    # labeling the Likert scale
                    if yd == y_data[-1]:
                        annotations.append(dict(xref='x', yref='paper',
                                                x=space + (xd[i]/2), y=1.1,
                                                text=top_labels[i],
                                                font=dict(family='Arial', size=14,
                                                          color='rgb(67, 67, 67)'),
                                                showarrow=False))
                    space += xd[i]

        layout['annotations'] = annotations

        #fig1 = go.Figure(data=traces, layout=layout)
        #py.iplot(fig,  filename='Results.png')
        #py.image.save_as(fig, filename='Results.png')
        #from IPython.display import Image
        #Image('Results.png')
        #py.image.ishow(fig)

    # Error Distributions
    all_cor = numpy.sum(df[df.apply(lambda x: min(x) == max(x), 1)][names[0]]==1)
    all_wro = numpy.sum(df[df.apply(lambda x: min(x) == max(x), 1)][names[0]]==0)
    disag = N - all_cor - all_wro
    print 'Predictions Distributions'
    print 'All correct : %0.2f  || Some correct : %0.2f || All wrong: %0.2f ' % \
                      (100*all_cor/float(N), 100*disag/float(N), 100*all_wro/float(N) )
    
    if print_flag:

        fig, ax = plt.subplots()
        s = plt.bar([1,2,3], [100*all_cor/float(N), 100*disag/float(N), 100*all_wro/float(N)], align='center', alpha=0.4)
        plt.xticks([1,2,3], ['All correct', 'Some correct', 'All wrong'])
        plt.ylabel('% Percentage of test dataset')
        plt.title('Ensemble Decisions')
        autolabel(s.patches, ax)
        plt.show()
    
    # Wrong Instances
    df_not_correct = df[df.sum(axis=1)!=L]
    N_wrong = df_not_correct.shape[0]
    counts = [all_wro*100/float(N)]
    for i in xrange(1,L):
        #print i
        if not(df_not_correct[df_not_correct.sum(axis=1)==i].empty):
            if df_not_correct[df_not_correct.sum(axis=1)==i].shape[0] == 0:
                counts.append(1*100/float(N))
            else:
                counts.append(df_not_correct[df_not_correct.sum(axis=1)==i].shape[0]*100/float(N))
        else:
            counts.append(float(0))
        #print counts
    non_corr_s = '%s : %0.2f  ||  ' % ('None Correct', counts[0])
    #print len(counts)
    for i in xrange(1,L):
        #print i
        non_corr_s +='%d correct : %0.2f  ||  ' % (i, counts[i])
    print 'Not all Correct Instances Distributions'
    print non_corr_s[:-4]
    if print_flag:
        fig, ax = plt.subplots()
        s = plt.bar([i for i in xrange(1,L+1)], counts, align='center', alpha=0.4)
        plt.xticks([i for i in xrange(1,L+1)], ['None correct']+['%d correct'% i for i in xrange(1,L)])
        plt.ylabel('% Percentage of not all correct instances')
        plt.title('Distribution of not all correct instances')
        autolabel(s.patches, ax)
        plt.show()

    return df, fig1


def autolabel(rects, ax):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2. ,1.0*height, '%0.2f' % float(height),ha='center', va='bottom')
