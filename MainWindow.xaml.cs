using Microsoft.Win32;
using Numpy;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Security.AccessControl;
using System.Security.Principal;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace NeuralNet
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        Network network;

        public MainWindow()
        {
            InitializeComponent();

            //readBtn.IsEnabled = false;

            network = new Network(new int[] { 784, 100, 10 });
        }

        private void myInkCanvas_MouseRightButtonUp(object sender, MouseButtonEventArgs e)
        {
            Matrix m = new Matrix();
            m.Scale(1.1d, 1.1d);
            ((InkCanvas)sender).Strokes.Transform(m, true);
        }

        private void trainBtn_Click(object sender, RoutedEventArgs e)
        {
            int epochs = int.Parse(numEpochs.Text);
            loadBtn.IsEnabled = false;
            (var trainingData, var validationData, var testData) = Loader.Load("data/mnist.npy");
            network.SGD(trainingData, epochs, 10, 3.0, testData);
            loadBtn.IsEnabled = true;
        }

        private void loadBtn_Click(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new OpenFileDialog();
            if ((bool)openFileDialog.ShowDialog())
            {
                try
                {
                    string file = openFileDialog.FileName;
                    network.Load(file);
                    readBtn.IsEnabled = true;
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Error: {ex.Message}");
                }
            }
        }

        private void readBtn_Click(object sender, RoutedEventArgs e)
        {
            double[] pixels = SaveCanvas();
            guessNumber.Content = "Number: " + network.ReadDigit(pixels);
        }

        public double[] SaveCanvas()
        {
            RenderTargetBitmap rtb = new RenderTargetBitmap((int)myInkCanvas.Width, (int)myInkCanvas.Height, 96d, 96d, PixelFormats.Default);
            rtb.Render(myInkCanvas);
            DrawingVisual dvInk = new DrawingVisual();
            DrawingContext dcInk = dvInk.RenderOpen();
            dcInk.DrawRectangle(myInkCanvas.Background, null, new Rect(0d, 0d, myInkCanvas.Width, myInkCanvas.Height));
            foreach (System.Windows.Ink.Stroke stroke in myInkCanvas.Strokes)
            {
                stroke.Draw(dcInk);
            }
            dcInk.Close();

            FileStream fs = File.Open("test.bmp", FileMode.OpenOrCreate);
            System.Windows.Media.Imaging.BmpBitmapEncoder encoder1 = new BmpBitmapEncoder();
            encoder1.Frames.Add(BitmapFrame.Create(rtb));
            encoder1.Save(fs);
            fs.Close();

            Bitmap bmpDest;
            using (Bitmap bmpOrig = new Bitmap("test.bmp"))
            {
                bmpDest = new Bitmap(bmpOrig, new System.Drawing.Size(28, 28));
                bmpDest.Save("test_small.bmp");
                Console.WriteLine(bmpDest.PixelFormat);
            }

            byte[] b = BitmapToByteArray(bmpDest);
            List<double> pixels = new List<double>();

            for (int i = 0; i <= b.Length - 4; i += 4)
            {
                Console.WriteLine("(" + b[i] + "," + b[i + 1] + "," + b[i + 2] + "," + b[i + 3] + ")");
                double val = 1.0 - (b[i] / 255.0);
                pixels.Add(val);
            }

            return pixels.ToArray();
        }

        public byte[] BitmapToByteArray(Bitmap bitmap)
        {
            System.Drawing.Rectangle rect = new System.Drawing.Rectangle(0, 0, bitmap.Width, bitmap.Height);
            BitmapData bitmapData = bitmap.LockBits(rect, ImageLockMode.ReadWrite, bitmap.PixelFormat);
            IntPtr ptr = bitmapData.Scan0;
            int size = Math.Abs(bitmapData.Stride) * bitmap.Height;
            byte[] rgb = new byte[size];
            System.Runtime.InteropServices.Marshal.Copy(ptr, rgb, 0, size);
            bitmap.UnlockBits(bitmapData);
            return rgb;
        }

        private void clearBtn_Click(object sender, RoutedEventArgs e)
        {
            myInkCanvas.Strokes.Clear();
        }
    }
}
