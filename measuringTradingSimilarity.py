import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import paired_distances
from scipy.stats import pearsonr


def collect(category, stock):
    fname = "outputFromKestut" + os.sep + stock
    df = pd.read_csv(fname)
    df1 = df[df['categories'] == category]
    df2 = df1[['trading_date', 'volume']]
    df_new = df2.groupby('trading_date').agg({'volume': 'sum'})
    # df_new.reset_index(level=0, inplace=True)
    df_new.columns = ['trading_date', 'volume']
    return df_new


def getAllTradingDates():
    AllDates = set()
    for stock in os.listdir("outputFromKestut"):
        fn = "outputFromKestut" + os.sep + stock
        df = pd.read_csv(fn)
        for date in df["trading_date"]:
            AllDates.add(date)
    print len(AllDates)
    return AllDates


def getTradingMatrixForOneCategory(category):
    AllDates = getAllTradingDates()
    directory = "outputJimmy" + os.sep + "category_" + str(category)
    listList = []
    Dfstocks = []
    for stock in os.listdir(directory):
        df = collect(category, stock)
        Dfstocks.append(df)
    for date in AllDates:
        list = []
        for i in range(0, len(Dfstocks)):
            df = Dfstocks[i]
            names = df.columns.values
            print names
            location = df.loc[df['trading_date'] == date]
            # volume=location.iloc[0]['volume']
            print location
        listList.append(list)


# getTradingMatrixForOneCategory(0)


def stupid_f():
    import glob

    files = glob.glob('./outputFromKestut/*')
    isin_list = map(lambda x: x[-16:-4], files)
    N = 100
    df = pd.concat([pd.read_csv(f, index_col=0) for f in files[:N]],
                   keys=isin_list[:N])
    df.reset_index(level=0, inplace=True)
    df.columns = ['ISIN', 'trading_date', 'volume', 'categories']
    df_agg = df.groupby(['categories', 'ISIN', 'trading_date']).volume.sum()

    df_unstacked = df_agg.unstack(2).fillna(0)
    A = df_unstacked.loc[0]
    B = df_unstacked.loc[23]
    AllDates = getAllTradingDates()
    cl = len(AllDates)
    l2 = len(A.index.values)
    df_fake = pd.DataFrame(np.random.randn(l2, cl), index=A.index)
    df_fake.columns = AllDates
    A = A.reindex_like(df_fake).fillna(0)
    B = B.reindex_like(df_fake).fillna(0)
    print A.shape[1]
    listList = []
    nr = len(A.index.values)
    AllDates = getAllTradingDates()
    cl = len(AllDates)
    df_fake = pd.DataFrame(np.random.randn(nr, cl), index=A.index)
    df_fake.columns = AllDates

    # df2 = pd.DataFrame(index=df1.index)
    for i in range(0, 110):
        for j in range(i, 110):
            A = df_unstacked.loc[i]
            B = df_unstacked.loc[j]
            A = A.reindex_like(df_fake).fillna(0)
            B = B.reindex_like(df_fake).fillna(0)
            listList.append(paired_distances(A.T, B.T))
    df_w = pd.DataFrame(listList)
    df.columns = AllDates
    df_w.to_csv('weighMatrix.csv')
    store = pd.HDFStore('./input/weightMatrix.h5')
    store['weighMatrix'] = df_w
    store.close()
    return df_w


def pearsonCor(groupA, groupB):
    lis = []
    for k in range(0, 1584):
        x = groupA.iloc[[k]].values
        y = groupB.iloc[[k]].values
        # x = scipy.array([-0.65499887,  2.34644428, 3.0])
        # y = scipy.array([-1.46049758,  3.86537321, 21.0])
        r_row, p_value = pearsonr(x[0], y[0])
        lis.append(r_row)
    return lis


def JaccardForOneDay(a, b):
    M01 = 0.0
    M10 = 0.0
    M11 = 0.0
    J = 0.0
    for k in range(0, 100):
        if a[k] * b[k] > 0:
            M11 = M11 + 1
        elif (a[k] != 0 and b[k] == 0):
            M10 = M10 + 1
        elif (a[k] == 0 and b[k] != 0):
            M01 = M01 + 1
    total = M11 + M01 + M10
    if total != 0:
        J = M11 / total
    return J


def JaccardForAlldates(groupA, groupB):
    lis = []
    for k in range(0, 1584):
        x = groupA.iloc[[k]].values
        y = groupB.iloc[[k]].values
        # x = scipy.array([-0.65499887,  2.34644428, 3.0])
        # y = scipy.array([-1.46049758,  3.86537321, 21.0])
        r_row = JaccardForOneDay(x[0], y[0])
        lis.append(r_row)
    return lis


def getWeightMatrixUsingJaccard():
    import glob
    import pandas as pd
    files = glob.glob('./outputFromKestut/*')
    isin_list = map(lambda x: x[-16:-4], files)
    N = 100
    df = pd.concat([pd.read_csv(f, index_col=0) for f in files[:N]],
                   keys=isin_list[:N])
    df.reset_index(level=0, inplace=True)
    df.columns = ['ISIN', 'trading_date', 'volume', 'categories']
    df_agg = df.groupby(['categories', 'ISIN', 'trading_date']).volume.sum()

    df_unstacked = df_agg.unstack(2).fillna(0)
    A0 = df_unstacked.loc[0]
    listList = []
    AllDates = getAllTradingDates()
    cl = len(AllDates)
    l2 = len(A0.index.values)
    df_fake = pd.DataFrame(np.random.randn(l2, cl), index=A0.index)
    df_fake.columns = AllDates

    for i in range(0, 110):
        for j in range(i + 1, 110):
            A = df_unstacked.loc[i]
            B = df_unstacked.loc[j]
            A = A.reindex_like(df_fake).fillna(0)
            B = B.reindex_like(df_fake).fillna(0)
            AT = A.T
            BT = B.T
            # similarity=paired_distances(AT,BT)
            # similarity=pearsonCor(AT,BT)
            lis = []
            print j
            for k in range(0, len(AT.index)):
                # print k
                x = AT.iloc[[k]].values
                y = BT.iloc[[k]].values
                r_row = JaccardForOneDay(x[0], y[0])
                lis.append(r_row)
            listList.append(lis)

    df_w = pd.DataFrame(listList)
    df_w.columns = AllDates
    df1 = df_w.T
    df1.to_csv('./output/jaccardWeighMatrix.csv')
    store = pd.HDFStore('./output/jaccardWeightMatrix.h5')
    store['jaccardWeighMatrix'] = df1
    store.close()
    return df_w


def getWeightMatrixUsingPearson():
    import glob
    import pandas as pd
    files = glob.glob('./outputFromKestut/*')
    isin_list = map(lambda x: x[-16:-4], files)
    N = 100
    df = pd.concat([pd.read_csv(f, index_col=0) for f in files[:N]],
                   keys=isin_list[:N])
    df.reset_index(level=0, inplace=True)
    df.columns = ['ISIN', 'trading_date', 'volume', 'categories']
    df_agg = df.groupby(['categories', 'ISIN', 'trading_date']).volume.sum()

    df_unstacked = df_agg.unstack(2).fillna(0)
    A0 = df_unstacked.loc[0]
    listList = []
    AllDates = getAllTradingDates()
    cl = len(AllDates)
    l2 = len(A0.index.values)
    df_fake = pd.DataFrame(np.random.randn(l2, cl), index=A0.index)
    df_fake.columns = AllDates

    for i in range(0, 110):
        for j in range(i + 1, 110):
            A = df_unstacked.loc[i]
            B = df_unstacked.loc[j]
            A = A.reindex_like(df_fake).fillna(0)
            B = B.reindex_like(df_fake).fillna(0)
            AT = A.T
            BT = B.T
            # similarity=paired_distances(AT,BT)
            # similarity=pearsonCor(AT,BT)
            lis = []
            print j
            for k in range(0, len(AT.index)):
                # print k
                x = AT.iloc[[k]].values
                y = BT.iloc[[k]].values
                # x = scipy.array([-0.65499887,  2.34644428, 3.0])
                # y = scipy.array([-1.46049758,  3.86537321, 21.0])
                # DataFrame.corr
                r_row, p_value = pearsonr(x[0], y[0])
                # print r_row
                lis.append(r_row)
            listList.append(lis)

    df_w = pd.DataFrame(listList)
    df_w.columns = AllDates
    df1 = df_w.T
    df1.to_csv('./output/pearsonWeighMatrix.csv')
    store = pd.HDFStore('./output/weightMatrixUsingPearson.h5')
    store['pearsonWeighMatrix'] = df1
    store.close()
    return df_w


# getWeightMatrixUsingPearson()

def getWeightMatrixUsingEuclideanORCosineSimilarity():
    import glob
    files = glob.glob('./outputFromKestut/*')
    isin_list = map(lambda x: x[-16:-4], files)
    N = 100
    df = pd.concat([pd.read_csv(f, index_col=0) for f in files[:N]],
                   keys=isin_list[:N])
    df.reset_index(level=0, inplace=True)
    df.columns = ['ISIN', 'trading_date', 'volume', 'categories']
    df_agg = df.groupby(['categories', 'ISIN', 'trading_date']).volume.sum()

    df_unstacked = df_agg.unstack(2).fillna(0)
    A0 = df_unstacked.loc[0]
    listList = []
    AllDates = getAllTradingDates()
    cl = len(AllDates)
    l2 = len(A0.index.values)
    df_fake = pd.DataFrame(np.random.randn(l2, cl), index=A0.index)
    df_fake.columns = AllDates

    for i in range(0, 110):
        for j in range(i + 1, 110):
            A = df_unstacked.loc[i]
            B = df_unstacked.loc[j]
            A = A.reindex_like(df_fake).fillna(0)
            B = B.reindex_like(df_fake).fillna(0)
            AT = A.T
            BT = B.T
            similarity = paired_distances(AT, BT, metric='cosine')
            listList.append(similarity)
    df_w = pd.DataFrame(listList)
    df_w.columns = AllDates
    df1 = df_w.T
    df1.to_csv('./output/weighMatrix.csv')
    store = pd.HDFStore('./output/weightMatrix.h5')
    store['weighMatrix'] = df1
    store.close()
    return df_w


getWeightMatrixUsingJaccard()
# getWeightMatrixUsingEuclideanORCosineSimilarity()
