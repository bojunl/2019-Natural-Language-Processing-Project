#!/usr/bin/env python
# coding: utf-8

# Export processed texts
def write_simple(li,filename):
    p = open(filename, "w")
    for alltext in li:
        alltext=html.unescape(alltext)
        alltext=alltext.rstrip()
        alltext = re.sub("\n", " ", alltext)
        alltext = re.sub(r"(^| )[0-9]+($| )", r" ", alltext)  # remove digits
        alltext = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",alltext)
        alltext = re.sub(r"\bhttp\S+\b", r"", alltext)
        alltext = re.sub(r"RT\s*@[^:]*:\s*.*",r" ",alltext)
        alltext = re.sub('#\S+', '', alltext)
        alltext = re.sub('RT|cc', '', alltext)
        alltext = re.sub("&amp; *", "", alltext)
        alltext = re.sub("&gt; *", "", alltext)
        alltext=alltext.rstrip()
        print(alltext.encode('utf8', 'ignore').decode('utf8'), file=p)
    p.close()
    
    
train_file="train.csv"
test_file="test.csv"



#write output to seperate files
positive_labels_training_texts=[]
negative_labels_training_texts=[]
zero_labels_training_texts=[]
all_training_text=[]
count=0
#index of target for example crude oil 5 min =4 30min=5 12hour=6
#gold 5min=7 30min=8 12hour =9
#usd index 5min=10 30min=11 12hours=12

indexNum=8
all_train_data=[]
with open(train_file, "U") as in_file1:
    reader1 = csv.reader(in_file1)
    for row1 in reader1:
        all_train_data.append(row1)
all_test_data=[]
with open(test_file, "U") as in_file1:
    reader1 = csv.reader(in_file1)
    for row1 in reader1:
        all_test_data.append(row1)

        

totalCount=0
totallen=0

for l in all_train_data:
    if count>=1:
        ind=int(l[indexNum])
        curtext=l[0]
        curlen=len(curtext.split())
        all_training_text.append(curtext)
        
        totallen+=curlen
        totalCount+=1
        if ind>0:
            positive_labels_training_texts.append(curtext)
        elif ind<0:
            negative_labels_training_texts.append(curtext)
        else:
            zero_labels_training_texts.append(curtext)
    count+=1
       
        
print("total length is "+str(totallen))
print("num tweets is "+str(totalCount))
print("avg is "+str(totallen/totalCount))


write_simple(positive_labels_training_texts,"postrain.txt")
write_simple(negative_labels_training_texts,"negtrain.txt")
write_simple(zero_labels_training_texts,"zerotrain.txt")
write_simple(all_training_text,"alltrain.txt")
print("done")




#Building Gensim model
import html
alltext = ""   
toksents = [] 


f = open("allTrump.txt", encoding="utf-8")

# read in the whole file
alltext = f.read().rstrip()
f.close()
alltext = html.unescape(alltext)
alltext = re.sub("\n", " ", alltext)
alltext = re.sub(r"(^| )[0-9]+($| )", r" ", alltext)  # remove digits
alltext = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",alltext)
alltext = re.sub(r"\bhttp\S+\b", r"", alltext)
alltext = re.sub(r"RT\s*@[^:]*:\s*.*",r" ",alltext)
alltext = re.sub('#\S+', '', alltext)
alltext = re.sub('RT|cc', '', alltext)
alltext = re.sub("&amp; *", "", alltext)
alltext = re.sub("&gt; *", "", alltext)
alltext = alltext.rstrip()
alltext=alltext.lower()

tok = nltk.sent_tokenize(alltext)
toksents=[]
for l in tok:
    w=nltk.word_tokenize(l)
    toksents.append(w)


alltokens = nltk.word_tokenize(alltext)
fdist = nltk.FreqDist(alltokens)


#remove some stop words for distribution graph
from nltk.corpus import stopwords
stoplist = stopwords.words('english')

newAllTokens=[]
for i in alltokens:
    if i not in stoplist:
        newAllTokens.append(i)
fdist = nltk.FreqDist(newAllTokens)
most_freq_words=[]
for i in fdist.most_common(10):
    most_freq_words.append(i[0])
print(most_freq_words)
fdist.plot(10, cumulative=False)


dimention=100
model = gensim.models.Word2Vec(toksents, size=dimention, window=5, min_count=5, workers=4)



#creating the vectors with cleaning the data
def create_vectors(filename,lines,tokens):
    f = open(filename)
    for l in f:
        l = l.rstrip()
        lines.append(l)    
        l = re.sub(r"(^| )[0-9]+($| )", r" ", l)  # remove digits
        l = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",l)
        l = re.sub(r"\bhttp\S+\b", r"", l)
        l = html.unescape(l)
        l = re.sub("\n", " ", l)
        l = re.sub(r"(^| )[0-9]+($| )", r" ", l)  # remove digits
        l = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",l)
        l = re.sub(r"\bhttp\S+\b", r"", l)
        l = re.sub(r"RT\s*@[^:]*:\s*.*",r" ",l)
        l = re.sub('#\S+', '', l)
        l = re.sub('RT|cc', '', l)
        l = re.sub("&amp; *", "", l)
        l = re.sub("&gt; *", "", l)
        l = l.rstrip()
        addme = [t.lower() for t in l.split()]
        tokens.append(addme)
    f.close()
    return tokens,lines



train_pos_vecs=[]
train_pos_lines=[]
train_pos_vecs,train_pos_lines=create_vectors("postrain.txt",train_pos_lines,train_pos_vecs)

train_neg_vecs=[]
train_neg_lines=[]
train_neg_vecs,train_neg_lines=create_vectors("negtrain.txt",train_neg_lines,train_neg_vecs)

train_zero_vecs=[]
train_zero_lines=[]
train_zero_vecs,train_zero_lines=create_vectors("zerotrain.txt",train_zero_lines,train_zero_vecs)
print("done")


#creating our own vectors
posvectors = []
for h in train_pos_vecs:
    totvec = np.zeros(dimention)
    for w in h:
        if w.lower() in model:
            totvec = totvec + model[w.lower()]
    posvectors.append(totvec)

negvectors = []       
for h in train_neg_vecs:
    totvec = np.zeros(dimention)
    for w in h:
        if w.lower() in model:
            totvec = totvec + model[w.lower()]
    negvectors.append(totvec)

print(len(train_neg_vecs))
print(len(negvectors))
print(len(negvectors[10]))

zerovectors = []       
for h in train_zero_vecs:
    totvec = np.zeros(dimention)
    for w in h:
        if w.lower() in model:
            totvec = totvec + model[w.lower()]
    zerovectors.append(totvec)


vecwords = []  
vecs = []    
for k in toksents:
    for a in k:
        if a in most_freq_words:
            try:
                kvec = model[a]
                vecs.append(kvec)
                vecwords.append(a)
            except:
                pass
    

vecs=vecs[:100]
pca = PCA(n_components=2, whiten=True)
vectors2d = pca.fit(vecs).transform(vecs)
for point, word in zip(vectors2d, vecwords):
    plt.scatter(point[0], point[1], c='b')
    plt.annotate(
        word, 
        xy=(point[0], point[1]),
        )



from scipy.spatial.distance import cdist
from sklearn import metrics


testtargets = []  
testvectors = []  

textlines = []    
textlinetoks = [] 
    
count=0
for l in all_test_data:
    if count>=1:
        ind=int(l[indexNum])
        curtext=l[0]
        testtargets.append(ind)
        addme=[t.lower() for t in curtext.split()]
        totvec = np.zeros(dimention)
        for h in addme:
            if h.lower() in model:
                totvec = totvec + model[h.lower()]
        testvectors.append(totvec)
    count+=1

print(len(testvectors))
print(len(testvectors[100]))
print(len(testtargets))

pos_d = sp.spatial.distance.cdist(testvectors, posvectors)
neg_d = sp.spatial.distance.cdist(testvectors, negvectors)
zero_d= sp.spatial.distance.cdist(testvectors, zerovectors)
posmins = pos_d.min(axis = 1)
negmins = neg_d.min(axis = 1)
zeromins= zero_d.min(axis =1)

predictedknn = []  
for item1, item2, item3 in zip(negmins, zeromins,posmins):
    if item1 < item2 and item1<item3:
        predictedknn.append(-1)
    elif item2<item1 and item2<item3:
        predictedknn.append(0)
    else:
        predictedknn.append(1)
        

alltargets = list(np.ones(len(posvectors)))
alltargets.extend(np.full_like(np.arange(len(negvectors)), -1))
alltargets.extend(np.zeros(len(zerovectors)))
alltargets = np.array(alltargets)
allvectors = posvectors +negvectors+ zerovectors 
print(len(allvectors[10]))
print(len(allvectors))
print(alltargets[:-5])


from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


#Naive Bayes
model = GaussianNB()
model.fit(allvectors, alltargets)
train = testtargets
test = model.predict(testvectors)
print("nb")
print(metrics.classification_report(train, test))

#SVM, KNN
svm = LinearSVC()
svm.fit(allvectors, alltargets)
svm_train = testtargets
svm_test = svm.predict(testvectors)
print("svm")
print(metrics.classification_report(svm_train, svm_test))
print("knn")
print(metrics.classification_report(testtargets, predictedknn))
