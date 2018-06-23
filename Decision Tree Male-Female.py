#Man-Woman Decision Tree
from sklearn import tree
from graphviz import Digraph
clf = tree.DecisionTreeClassifier() 

#[height cm, hair-length cm, voice-pitch (0-low, 1-high)]                                             
X = [ [180, 15,0],                                                              
      [167, 42,1],                                                              
      [136, 35,1],                                                              
      [174, 15,0],                                                              
      [141, 28,1]]                                                              

Y = ['man', 'woman', 'woman', 'man', 'woman']

clf = clf.fit(X, Y)                                                             
prediction = clf.predict([[150, 50,1]])                                         
print(prediction) 
tree.export_graphviz(clf,
    out_file='tree1.dot')     
