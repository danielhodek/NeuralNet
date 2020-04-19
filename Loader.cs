using Numpy;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    class Loader
    {
        private static (NDarray, NDarray, NDarray) LoadData(string path)
        {
            NDarray data = np.load(path, null, true, true);
            NDarray trainingData = data[0];
            NDarray validationData = data[1];
            NDarray testData = data[2];
            return (trainingData, validationData, testData);
        }

        public static (List<NDarray>, List<NDarray>, List<NDarray>) Load(string path)
        {
            (var trD, var vaD, var teD) = LoadData(path);

            List<NDarray> trainingInputs = new List<NDarray>();
            List<NDarray> trainingResults = new List<NDarray>();
            List<NDarray> validationInputs = new List<NDarray>();
            List<NDarray> testInputs = new List<NDarray>();

            for (int x = 0; x < trD[1].size; x++)
                trainingInputs.Add(np.reshape(trD[0][x], new int[] { 784, 1 }));
            for (int y = 0; y < trD[1].size; y++)
                trainingResults.Add(VectorizedResults(trD[1][y]));
            List<NDarray> trainingData = Helper.Zip(trainingInputs, trainingResults);

            for (int x = 0; x < vaD[1].size; x++)
                validationInputs.Add(np.reshape(vaD[0][x], new int[] { 784, 1 }));
            List<NDarray> validationData = Helper.Zip(validationInputs, vaD[1]);

            for (int x = 0; x < teD[1].size; x++)
                testInputs.Add(np.reshape(teD[0][x], new int[] { 784, 1 }));
            List<NDarray> testData = Helper.Zip(testInputs, teD[1]);

            return (trainingData, validationData, testData);
        }

        private static NDarray VectorizedResults(NDarray j)
        {
            NDarray e = np.zeros(new int[] { 10, 1 });
            e[j] = np.ones(new int[] { 1 });
            return e;
        }
    }
}
