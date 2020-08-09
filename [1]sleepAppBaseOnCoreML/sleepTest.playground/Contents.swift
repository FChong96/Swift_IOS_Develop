import CreateML
import Foundation

//read json
let data = try MLDataTable(contentsOf: URL(fileURLWithPath: "/Users/fengchong/Desktop/coding/better-rest.json"))
//split data: training data(validation data)/testing data
let (trainingData, testingData) = data.randomSplit(by: 0.2)//80%training data

let regressor = try MLRegressor(trainingData: trainingData, targetColumn: "actualSleep") //regressor (回归器) set , targetColumn from better-rest.json

let evaluationMatrics = regressor.evaluation(on: testingData)

print(evaluationMatrics.rootMeanSquaredError) //Mean Squared Error （均方差）
print(evaluationMatrics.maximumError)

let metadata = MLModelMetadata(author: "Feng Chong", shortDescription: "a model trained to predict optimum sleep times for coffee drinkers.", version: "1.0")

try regressor.write(to: URL(fileURLWithPath: "/Users/fengchong/Desktop/coding/sleepCalculatorTest.mlmodel"), metadata: metadata)//this sentence should re-run the create ml model
