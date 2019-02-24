"""Demonstrates how to make a simple call to the Natural Language API."""
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
#from sklearn.cluster import DBSCAN
#from sklearn import metrics
#from sklearn.decomposition import PCA
import argparse
#%matplotlib inline

from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types


def print_result(annotations):
    score = annotations.document_sentiment.score
    magnitude = annotations.document_sentiment.magnitude

    for index, sentence in enumerate(annotations.sentences):
        sentence_sentiment = sentence.sentiment.score
        print('Sentence {} has a sentiment score of {}'.format(
            index, sentence_sentiment))

    print('Overall Sentiment: score of {} with magnitude of {}'.format(
        score, magnitude))
    return 0


def analyze(movie_review_filename, outfilename, start_row = -1, end_row = -1):
    """Run a sentiment analysis request on text within a passed filename."""
    client = language.LanguageServiceClient()
    df = pd.read_csv(movie_review_filename)
    if start_row < 0:
        start_row, end_row = 0 , len(df)
    if end_row < 1 or end_row > len(df) or end_row < start_row:
        end_row = len(df)
#    print(start_row,end_row,outfilename)
#    print(len(df))
    
    dfuse = df.sample(n=end_row-start_row, random_state=1)
    scores = []
    mags = []
    msgs = []
#    cnt = 0
    for row in range(end_row-start_row):
        i = dfuse.iloc[row,1]
#        cnt += 1
#        if not cnt%1000:
#            print(cnt)
#        print(i)
        doc = types.Document(
                content=i,
                type=enums.Document.Type.PLAIN_TEXT)
        try:
            annotations = client.analyze_sentiment(document=doc)
            message = "None"
        except:
            annotations.document_sentiment.score = 0
            annotations.document_sentiment.magnitude = -1
            message = "Error occurred"
        scores.append(annotations.document_sentiment.score)
        mags.append(annotations.document_sentiment.magnitude)
        msgs.append(message)
    
    dfuse['Score'] = 0
    dfuse.iloc[:,2] = scores
    dfuse['Magnitude'] = 0
    dfuse.iloc[:,3] = mags
    dfuse['Message'] = ""
    dfuse.iloc[:,4] = msgs
    dfuse.to_csv(outfilename,index = False)
    dfplot = dfuse.iloc[start_row:end_row,:]
    dfplot[dfplot['Magnitude'] > -1].plot('Score','Magnitude',kind = 'scatter')
    plt.title("Randomized 100 Samples, Percent of Missing = "+str(float(len(dfplot[dfplot['Magnitude']==-1]))/len(dfplot)*100)+"%")
    plt.show()
    
    
#    dbscan = DBSCAN(eps=.05).fit(dfplot)
#    core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
#    core_samples_mask[dbscan.core_sample_indices_] = True
#    labels = dbscan.labels_
#    
#    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#    n_noise_ = list(labels).count(-1)
#    
#    print('Estimated number of clusters: %d' % n_clusters_)
#    print('Estimated number of noise points: %d' % n_noise_)
##    print("Silhouette Coefficient: %0.3f"
##          % metrics.silhouette_score(dfdb, labels))
#    
#    unique_labels = set(labels)
#    colors = [plt.cm.Spectral(each)
#        for each in np.linspace(0, 1, len(unique_labels))]
#    for k, col in zip(unique_labels, colors):
#        if k == -1:
#            # Black used for noise.
#            col = [0, 0, 0, 1]
#    
#        class_member_mask = (labels == k)
#    
#        xy = dfplot[class_member_mask & core_samples_mask]
#        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#                 markeredgecolor='k', markersize=14)
#    
#        xy = dfplot[class_member_mask & ~core_samples_mask]
#        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#                 markeredgecolor='k', markersize=6)
#    
#    plt.title('Estimated number of clusters: %d' % n_clusters_)
#    plt.show()
#    ws = wb['test']
#    ws['C'+str(start_row-1)] = "Score"
#    ws['D'+str(start_row-1)] = "Magnitude"
#    for row in range(start_row,end_row+1):
#        temp = ws['B'+str(row)]
#        doc = types.Document(
#            content=temp,
#            type=enums.Document.Type.PLAIN_TEXT)
#        annotations = client.analyze_sentiment(document=doc)
#        ws['C'+str(row)] = annotations.document_sentiment.score
#        ws['D'+str(row)] = annotations.document_sentiment.magnitude
#        
#    wb.save(outfilename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'movie_review_filename',
        help='The filename of the movie review you\'d like to analyze.')
    parser.add_argument(
        'row_start',
        help='The first row in the Excel file to analyze.')
    parser.add_argument(
        'row_end',
        help='The last row in the Excel file to analyze.')
    parser.add_argument(
        'output_filename',
        help='The filename of the output Excel file.')
    args = parser.parse_args()

    analyze(args.movie_review_filename,args.output_filename,int(args.row_start),int(args.row_end))
