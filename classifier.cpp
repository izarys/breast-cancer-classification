#include <bits/stdc++.h>
using namespace std;

const int N = 699; // the number of data instances in total
const int M = 9; // the number of attributes
const int N_TRAINING = 499;
const int N_TESTING = 200;
const int K = 13; // parameter for k-NN 

struct DataInstance {
    int id;
    vector<int> attribute;
    int result;
    
    void read() {
        cin >> id;
        attribute.resize(M);
        for(int i = 0; i < M; i++) {
            cin >> attribute[i];
        }
        cin >> result;
    }
};


int getArithmeticMean(vector<DataInstance>& data, int i) {
    int arithmeticMean = 0;
    int numberOfValues = 0;
    for(int j = 0; j < N; j++) {
        if(data[j].attribute[i] != -1) {
            arithmeticMean += data[j].attribute[i];
            numberOfValues++;
        }
    }

    return arithmeticMean / numberOfValues;
}

void fillInMissingData(vector<DataInstance>& data) {
    for(int i = 0; i < M; i++) {
        int arithmeticMean = getArithmeticMean(data, i);
        for(int j = 0; j < N; j++) {
            if(data[j].attribute[i] == -1) {
                data[j].attribute[i] = arithmeticMean;
            }
        }
    }
}

void splitDataSet(vector<DataInstance>& data, vector<DataInstance>& trainingData, vector<DataInstance>& testingData) {
    random_shuffle(data.begin(), data.end());
    trainingData = vector<DataInstance> (data.begin(), data.begin() + N_TRAINING);
    testingData = vector<DataInstance> (data.begin() + N_TRAINING, data.end()); 
}

class Classifier {
    public:
    vector<DataInstance> trainingData, testingData;
    virtual void train() = 0;
    virtual int classify(DataInstance) = 0;
    virtual void test() = 0;
};

class NearestNeighborClassifier : Classifier {    
    vector<pair<int,int>> computeDistances(DataInstance dataInstance) {
        vector<pair<int,int>> dist(N_TRAINING);
        for(int i = 0; i < N_TRAINING; i++) {
            int d = 0;

            for(int j = 0; j < M; j++) {
                int tmp = dataInstance.attribute[j] - trainingData[i].attribute[j];
                d += tmp * tmp;
            }
            dist[i] = {d, i};
        }
        return dist;
    }

    int getResult(vector<pair<int,int>>& dist) {
        int benignCount = 0;
        int malignantCount = 0;

        for(int i = 0; i < K; i++) {
            if(trainingData[dist[i].second].result == 2) {
                benignCount++;
            }
            else {
                malignantCount++;
            }
        }

        return benignCount > malignantCount ? 2 : 4;
    }
    
    public:
    NearestNeighborClassifier(vector<DataInstance>& trainingData, vector<DataInstance>& testingData) {
        this -> trainingData = trainingData;
        this -> testingData = testingData;
    }

    void train() {}

    int classify(DataInstance dataInstance) {
        vector<pair<int,int>> dist = computeDistances(dataInstance);
        sort(dist.begin(), dist.end());
        return getResult(dist);    
    }

    void test() {
        train();
        int truePositive = 0, falsePositive = 0;
        int trueNegative = 0, falseNegative = 0;
        
        for(int i = 0; i < N_TESTING; i++) {
            int result = classify(testingData[i]);
            
            if(result == 4) {
                if(testingData[i].result == result) {
                    truePositive++;
                } else {
                    falseNegative++;
                }
            } else {
                if(testingData[i].result == result) {
                    trueNegative++;
                } else {
                    falsePositive++;
                }
            }
        }

        cout << "TP: " << truePositive << endl;
        cout << "TN: " << trueNegative << endl;
        cout << "FP: " << falsePositive << endl;
        cout << "FN: " << falseNegative << endl;
    }
};

class NaiveBayesClassifier : Classifier {
    vector<vector<double>> benignCP; // benignCP[i][j] - the conditional probability of i-th attribute
                                     // being equal to "j" given that the class is "benign"
    vector<vector<double>> malignantCP; // malignantCP[i][j] - the conditional probability of i-th attribute
                                     // being equal to "j" given that the class is "malignant"
    double benignProbability; // the probability of class being "benign"
    double malignantProbability; // the probability of class being "malignant"

    public:
    NaiveBayesClassifier(vector<DataInstance>& trainingData, vector<DataInstance>& testingData) {
        this -> trainingData = trainingData;
        this -> testingData = testingData;
        benignCP.resize(M, vector<double> (11,0));
        malignantCP.resize(M, vector<double> (11,0));
    }
    
    void train() {
        int benignCount = 0, malignantCount = 0;
        for(int i = 0; i < N_TRAINING; i++) {
            if(trainingData[i].result == 2) {
                benignCount++;
            }
            else {
                malignantCount++;
            }
            // counting the occurrences of attribute values
            // given that the class is benign / malignant
            for(int j = 0; j < M; j++) {
                if(trainingData[i].result == 2) {
                    benignCP[j][trainingData[i].attribute[j]]++;
                }
                else {
                    malignantCP[j][trainingData[i].attribute[j]]++;
                }
            }
        }

        for(int i = 0; i < M; i++) {          
            // counting the number of benign / malignant occurences of i-th attribute
            int tmpB = 0, tmpM = 0;
            for(int j = 1; j <= 10; j++) {
                tmpB += benignCP[i][j];
                tmpM += malignantCP[i][j];
            }

            // dividing the number of occurrences of an attribute value
            // by the number of benign / malignant occurences of i-th attribute
            // to get the conditional probability
            // of i-th attribute having a value of "j" given that the class was benign / malignant
            for(int j = 1; j <= 10; j++) {
                benignCP[i][j] /= (double)tmpB;
                malignantCP[i][j] /= (double)tmpM;
            }

        }

        benignProbability = ((double)benignCount / (double)(benignCount + malignantCount));
        malignantProbability = ((double) malignantCount / (double)(benignCount + malignantCount));
    }

    int classify(DataInstance dataInstance) {
        double totalBenignProbability = benignProbability;      
        double totalMalignantProbability = malignantProbability;

        for(int i = 0; i < M; i++) {
            totalBenignProbability *= benignCP[i][dataInstance.attribute[i]];
            totalMalignantProbability *= malignantCP[i][dataInstance.attribute[i]];
        }

        return totalBenignProbability > totalMalignantProbability ? 2 : 4;
    }

    void test() {
        train();
        int truePositive = 0, falsePositive = 0;
        int trueNegative = 0, falseNegative = 0;
        
        for(int i = 0; i < N_TESTING; i++) {
            int result = classify(testingData[i]);
            
            if(result == 4) {
                if(testingData[i].result == result) {
                    truePositive++;
                } else {
                    falseNegative++;
                }
            } else {
                if(testingData[i].result == result) {
                    trueNegative++;
                } else {
                    falsePositive++;
                }
            }
        }

        cout << "TP: " << truePositive << endl;
        cout << "TN: " << trueNegative << endl;
        cout << "FP: " << falsePositive << endl;
        cout << "FN: " << falseNegative << endl;
    }
};

int main() {

    vector<DataInstance> data;
    data.resize(N);

    for(int i = 0; i < N; i++) {
        data[i].read();
    }

    fillInMissingData(data);
    vector<DataInstance> trainingData, testingData;

    splitDataSet(data, trainingData, testingData);

    NearestNeighborClassifier NN (trainingData, testingData);
    cout << "k-Nearest Neighbor Classifier:" << endl;
    NN.test();

    NaiveBayesClassifier NB (trainingData, testingData);
    cout << "Naive Bayes Classifier:" << endl;
    NB.test();
}