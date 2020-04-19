using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Numpy;

namespace NeuralNet
{
    class Network
    {
        private int numLayers;
        private List<NDarray> biases = new List<NDarray>();
        private List<NDarray> weights = new List<NDarray>();

        public Network(int[] sizes)
        {
            numLayers = sizes.Length;

            for (int y = 1; y < numLayers; y++)
                biases.Add(np.random.randn(sizes[y], 1));

            for (int x = 0, y = 1; y < numLayers; x++, y++)
                weights.Add(np.random.randn(sizes[y], sizes[x]));
        }

        public void Load(string file)
        {
            NDarray data = np.load(file);
            NDarray b = data[0];
            NDarray w = data[1];
            List<NDarray> btmp = new List<NDarray>();
            List<NDarray> wtmp = new List<NDarray>();
            for (int i = 0; i < b.size; i++)
                btmp.Add(b[i]);
            for (int i = 0; i < w.size; i++)
                wtmp.Add(w[i]);
            biases = btmp;
            weights = wtmp;
        }

        public void Save()
        {
            List<NDarray> tmp = new List<NDarray>();
            NDarray b = np.array(biases);
            NDarray w = np.array(weights);
            tmp.Add(b);
            tmp.Add(w);
            NDarray data = np.array(tmp);
            np.save("network", data);
        }

        public string ReadDigit(double[] pixels)
        {
            NDarray input = np.array(pixels);
            Console.WriteLine(input);
            input = np.reshape(input, new int[] { 784, 1 });
            NDarray output = FeedForward(input);
            var x = np.argmax(output);
            return x.str;
        }

        public NDarray FeedForward(NDarray a)
        {
            for (int i = 0; i < weights.Count; i++)
                a = Sigmoid(np.dot(weights[i], a) + biases[i]);
            return a;
        }

        public void SGD(List<NDarray> trainingData, int epochs, int miniBatchSize, double eta, List<NDarray> testData = null)
        {
            int nTest = 0;

            if (testData != null)
                nTest = testData.Count;

            int n = trainingData.Count;
            List<List<NDarray>> miniBatches = new List<List<NDarray>>();

            for (int i = 0; i < epochs; i++)
            {
                np.random.shuffle(np.array(trainingData));
                int count = 0;
                List<NDarray> miniBatch = new List<NDarray>();

                for (int j = 0; j < n; j++)
                {
                    miniBatch.Add(trainingData[j]);

                    if (++count == miniBatchSize)
                    {
                        miniBatches.Add(miniBatch);
                        miniBatch = new List<NDarray>();
                        count = 0;
                    }
                }

                foreach (var mb in miniBatches)
                    UpdateMiniBatch(mb, eta);

                if (testData != null)
                    Console.WriteLine("Epoch {0}: {1} / {2}", i, Evaluate(testData), nTest);
            }

            Save();
        }

        private NDarray Sigmoid(NDarray z)
        {
            return 1.0 / (1.0 + np.exp(-z));
        }

        private NDarray SigmoidPrime(NDarray z)
        {
            return Sigmoid(z) * (-Sigmoid(z) + 1.0);
        }

        private NDarray CostDerivative(NDarray outputActivations, NDarray y)
        {
            return outputActivations - y;
        }

        private void UpdateMiniBatch(List<NDarray> miniBatch, double eta)
        {
            List<NDarray> nablaB = new List<NDarray>();
            List<NDarray> nablaW = new List<NDarray>();

            foreach (var b in biases)
                nablaB.Add(np.zeros(b.shape));

            foreach (var w in weights)
                nablaW.Add(np.zeros(w.shape));

            foreach (var tuple in miniBatch)
            {
                (List<NDarray> deltaNableB, List<NDarray> deltaNableW) = Backprop(tuple[0], tuple[1]);

                for (int i = 0; i < nablaB.Count; i++)
                    nablaB[i] = nablaB[i] + deltaNableB[i];

                for (int i = 0; i < nablaW.Count; i++)
                    nablaW[i] = nablaW[i] + deltaNableW[i];
            }

            for (int i = 0; i < weights.Count; i++)
                weights[i] = weights[i] - (eta / miniBatch.Count) * nablaW[i];

            for (int i = 0; i < biases.Count; i++)
                biases[i] = biases[i] - (eta / miniBatch.Count) * nablaB[i];
        }

        private int Evaluate(List<NDarray> testData)
        {
            List<NDarray> testResults = new List<NDarray>();

            foreach (var tuple in testData)
            {
                List<NDarray> tmp = new List<NDarray>();
                tmp.Add(np.argmax(FeedForward(tuple[0])));
                tmp.Add(tuple[1]);
                testResults.Add(np.array(tmp));
            }
            
            int sum = 0;

            foreach (var tuple in testResults)
                if (np.array_equal(tuple[0], tuple[1]))
                    ++sum;

            return sum;
        }

        private (List<NDarray>, List<NDarray>) Backprop(NDarray x, NDarray y)
        {
            List<NDarray> nablaB = new List<NDarray>();
            List<NDarray> nablaW = new List<NDarray>();
                
            for (int i = 0; i < biases.Count; i++)
            {
                nablaB.Add(np.zeros(biases[i].shape));
                nablaW.Add(np.zeros(weights[i].shape));
            }                

            // feed forward
            NDarray activation = x;
            List<NDarray> activations = new List<NDarray>();
            activation = np.reshape(activation, new int[] { 784, 1 });
            activations.Add(x);
            List<NDarray> zs = new List<NDarray>();

            for (int i = 0; i < biases.Count; i++)
            {
                NDarray z = np.dot(weights[i], activation) + biases[i];
                zs.Add(z);
                activation = Sigmoid(z);
                activations.Add(activation);
            }

            // backwards pass
            y = np.reshape(y, new int[] { 10, 1 });
            NDarray delta = CostDerivative(activations[activations.Count - 1], y) * SigmoidPrime(zs[zs.Count - 1]);
            nablaB[nablaB.Count - 1] = delta;
            nablaW[nablaW.Count - 1] = np.dot(delta, np.transpose(activations[activations.Count - 2]));

            for (int i = 2; i < numLayers; i++)
            {
                NDarray z = zs[zs.Count - i];
                NDarray sp = SigmoidPrime(z);
                delta = np.dot(np.transpose(weights[weights.Count - i + 1]), delta) * sp;
                nablaB[nablaB.Count - i] = delta;
                nablaW[nablaW.Count - i] = np.dot(delta, np.transpose(np.reshape(activations[activations.Count - i - 1], new int[] { 784, 1 })));
            }   

            return (nablaB, nablaW);
        }
    }
}
