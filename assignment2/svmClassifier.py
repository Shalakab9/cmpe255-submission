from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score,f1_score 
from sklearn.metrics import precision_recall_fscore_support

class FaceRecognition:

    def load_data(self):
        faces = fetch_lfw_people(min_faces_per_person=60)
        print('data loaded')
        print(faces.target_names)
        print(faces.images.shape)
        return faces

    def init(self,faces):
        n_samples, h, w = faces.images.shape
        x=faces.data
        n_features = x.shape[1]
        y = faces.target
        target_names = faces.target_names
        n_classes = target_names.shape[0]
        print("Total dataset size:")
        print("n_samples: %d" % n_samples)
        print("n_features: %d" % n_features)
        print("n_classes: %d" % n_classes)
        return x,y,target_names,h,w

    def defineModel(self):
        pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
        svc = SVC(kernel='rbf', class_weight='balanced')
        model = make_pipeline(pca, svc)
        return model

    def splitData(self,x,y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)
        return x_train, x_test, y_train, y_test

    def gridSearchCV(self,model,x_train, x_test, y_train, y_test):
        param_grid = {'pca__n_components': [5, 15, 30, 45, 64],
                    'svc__C': [1,5,10,50],
                    'svc__gamma' :['scale', 'auto']}
        clf = GridSearchCV(model, param_grid, n_jobs=-1)
        best=clf.fit(x_train, y_train)
        y_pred = best.predict(x_test)
        return y_pred

    def precisionScore(self,y_test, y_pred):
        return precision_score(y_test, y_pred, average='macro')

    def recallScore(self,y_test, y_pred):
        return recall_score(y_test, y_pred, average='macro')

    def f1Score(self,y_test, y_pred):
        return f1_score(y_test, y_pred, average='macro')

    def support(self,y_test, y_pred):
        return precision_recall_fscore_support(y_test, y_pred, average=None)[3]

    def title(self,y_pred, y_test, target_names, i):
        pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
        true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
        return pred_name, true_name

    def predictTitle(self,prediction_titles):
        actual=[]
        predicted=[]
        for element in prediction_titles:
            predicted.append(element[0])
            actual.append(element[1])
        return actual,predicted

    def plot_gallery(self,images, actual, predicted, h, w, n_row=4, n_col=6):
    
        plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            if actual[i]==predicted[i]: 
                plt.title(predicted[i], size=12, fontdict={'color': 'black'})
            else:
                plt.title(predicted[i], size=12, fontdict={'color': 'red'})
            plt.xticks(())
            plt.yticks(())
        plt.show()

    def confusionMatrix(self,y_test, y_pred,target_names):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8,6))
        ax= plt.subplot()
        sns.heatmap(cm.T, annot=True, fmt='g', ax=ax)
        ax.xaxis.set_ticklabels(target_names, rotation=90)
        ax.yaxis.set_ticklabels(target_names, rotation=0)
        ax.set_ylabel('Predicted')
        ax.set_xlabel('Actual')
        plt.show()

        

def main():
    faceRecognition=FaceRecognition()
    faces=faceRecognition.load_data()
    x,y,target_names,h,w=faceRecognition.init(faces)
    model=faceRecognition.defineModel()
    x_train, x_test, y_train, y_test=faceRecognition.splitData(x,y)
    y_pred=faceRecognition.gridSearchCV(model,x_train, x_test, y_train, y_test)
    precision_score=faceRecognition.precisionScore(y_test, y_pred)
    recall_score=faceRecognition.recallScore(y_test, y_pred)
    f1_score=faceRecognition.f1Score(y_test, y_pred)
    support=faceRecognition.support(y_test, y_pred)
    print("Precision score: ",precision_score)
    print("Recall Score : ",recall_score )
    print("F1 Score : ",f1_score )
    print("Support : ",support )

    prediction_titles = [faceRecognition.title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]
    actual,predicted=faceRecognition.predictTitle(prediction_titles)
    faceRecognition.plot_gallery(x_test, actual, predicted, h, w)
    faceRecognition.confusionMatrix(y_test, y_pred,target_names)


if __name__ == "__main__":
    main()