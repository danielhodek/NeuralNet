using Numpy;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    class Helper
    {
        public static List<NDarray> Zip(List<NDarray> arrs, NDarray arr)
        {
            List<NDarray> data = new List<NDarray>();
            int min = Math.Min(arrs.Count, arr.size);

            for (int i = 0; i < min; i++)
            {
                List<NDarray> tmp = new List<NDarray>();
                tmp.Add(arrs[i]);
                tmp.Add(arr[i]);
                NDarray ndarr = np.array(tmp);
                data.Add(ndarr);
            }
                
            return data;
        }

        public static List<NDarray> Zip(List<NDarray> arrs1, List<NDarray> arrs2)
        {
            List<NDarray> data = new List<NDarray>();
            int min = Math.Min(arrs1.Count, arrs2.Count);

            for (int i = 0; i < min; i++)
            {
                List<NDarray> tmp = new List<NDarray>();
                tmp.Add(arrs1[i]);
                tmp.Add(arrs2[i]);
                NDarray ndarr = np.array(tmp);
                data.Add(ndarr);
            }

            return data;
        }
    }
}
