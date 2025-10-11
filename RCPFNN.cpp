/*
author : @rebwar_ai
*/
#include <iostream>
#include <vector>
#include <stdexcept>
#include "Layer.hpp"
#include "NeuralNetwork.hpp"
#include "CSVLoader.hpp"
#include "Log.hpp"
#include <sstream>

using namespace std;

int main(){

    try{
        vector<Layer> layers;

        
        layers.emplace_back(0,24,ActivationType::None);
        layers.emplace_back(1, 12, ActivationType::ReLU);
        layers.emplace_back(2, 1, ActivationType::Sigmoid);

        NeuralNetwork nn(layers);

        vector<vector<double>> training_features;
        vector<vector<double>> training_labels;
        vector<int> training_ids;

        vector<vector<double>> test_features;
        vector<vector<double>> test_labels;
        vector<int> test_ids;

        string load_model;

        if (CSV::loadAndSplitSensorData("sensor_readings_24.csv",
                                    training_features, training_labels, training_ids,
                                    test_features, test_labels, test_ids,0.8)) {
            
            cout << "Do you want to load the model (y/n): ";
            cin >> load_model;
            if (load_model == "y" || load_model == "Y") {
                // load model
                cout << "Loading model...\n";
                nn.loadModel();
            } else {
                load_model = "n";
                nn.train(training_features, training_labels, 0.029, 1300,8);
            }                          
            

        }else{
            cerr << "Failed to load sensor data !\n";
            return 1; 
        }
        
        
        cout << "-------------------Predictions--------------------\n";
        L::log("-------------------Predictions--------------------\n");
        for (size_t i = 0; i < test_features.size(); ++i) {
            stringstream data;
            data << "-------------------Prediction["<<i<<"]--------------------\n"; 
            data << "Row ID     : " << test_ids[i] << "\n";

            data << "Sensor Data: ";
            for (double val : test_features[i]) {
                data << fixed << setprecision(2) << val << " ";
            }
            data << "\n";

            vector<double> prediction = nn.predict(test_features[i]);

            data << "Prediction : " << fixed << setprecision(4) << prediction[0] << "\n";
            data << "Actual     : " << test_labels[i][0] << "\n";

            cout << data.str();
            L::log(data.str());
        }

        int tp = 0, tn = 0, fp = 0, fn = 0;

        for (size_t i = 0; i < test_features.size(); ++i) {
            double pred = nn.predict(test_features[i])[0];
            int predicted = pred >= 0.5 ? 1 : 0;
            int actual = static_cast<int>(test_labels[i][0]);

            if (predicted == 1 && actual == 1) tp++;
            else if (predicted == 0 && actual == 0) tn++;
            else if (predicted == 1 && actual == 0) fp++;
            else if (predicted == 0 && actual == 1) fn++;
        }

        int total = tp + tn + fp + fn;
        double accuracy = static_cast<double>(tp + tn) / total;
        double precision = tp + fp == 0 ? 0.0 : static_cast<double>(tp) / (tp + fp);
        double recall    = tp + fn == 0 ? 0.0 : static_cast<double>(tp) / (tp + fn);
        double f1_score  = (precision + recall) == 0 ? 0.0 :
            2.0 * (precision * recall) / (precision + recall);

        stringstream data;
        // Display results
        data << "-------------------Metrics---------------------\n";
        data << "\nConfusion Matrix:\n";
        data << "TP: " << tp << " | FP: " << fp << "\n";
        data << "FN: " << fn << " | TN: " << tn << "\n";

        data << std::fixed << std::setprecision(4);
        data << "\nAccuracy : " << accuracy * 100 << "%\n";
        data << "Precision: " << precision * 100 << "%\n";
        data << "Recall   : " << recall * 100 << "%\n";
        data << "F1 Score : " << f1_score * 100 << "%\n";
        cout << data.str();
        L::log(data.str());

        if(load_model == "n"){
            string save_model;
            cout << "Do you want to save the model (y/n): ";
            cin >> save_model;
            if (save_model == "y" || save_model == "Y") {
                // Save model
                cout << "Saving model...\n";
                nn.saveModel();
            } else {
                cout << "Model not saved.\n";
            }
        }
        
    }catch(const exception &e){
        cerr << " ERROR : " << e.what() << endl;
        return 1;
    }

    return 0;
}

// layers.emplace_back(0,24,ActivationType::None);
// layers.emplace_back(1, 64, ActivationType::ReLU);
// layers.emplace_back(2, 32, ActivationType::ReLU);
// layers.emplace_back(3, 16, ActivationType::ReLU);

// layers.emplace_back(4, 1, ActivationType::Sigmoid);
