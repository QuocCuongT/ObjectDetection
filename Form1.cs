using Emgu.CV.Dnn;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;

using System.Windows.Forms;
using NAudio.Wave;

using Emgu.CV;
using Emgu.CV.Structure;
using System.IO;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;
using Emgu.CV.UI;
using Emgu.CV.Dnn;
using System.Threading;


namespace object_Detection_Test
{
    public partial class Form1 : Form
    {
        // Yolo3
        Net Model = null;
        UInt16 Temp = 0;
        
        Image<Bgr, byte> img = null;
        List<string> ClassLabels = null;

        // Camera
        VideoCapture capture = null;
        VideoCapture video = null;
        bool captureInProgress = false;
        int cameradevice = 0;
        Mat frame;

        // Biến test
        object sender;
        EventArgs e;

        // Image
        bool Image_count = false;
        string outputDirectory;

        //Video
        WaveOut waveOut;
        AudioFileReader AudioReader;
        //VideoCapture _capture = null;
        IBackgroundSubtractor backgroundSubtractor;

        public Form1()
        {
            InitializeComponent();
        }

        private void readModelToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                OpenFileDialog dialog = new OpenFileDialog();
                dialog.Multiselect = true;
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    var PathWeights = dialog.FileNames.Where(x => x.ToLower().EndsWith(".weights")).First();
                    var Pathconfig = dialog.FileNames.Where(x => x.ToLower().EndsWith(".cfg")).First();
                    var PathClasses = dialog.FileNames.Where(x => x.ToLower().EndsWith(".names")).First();

                    Model = DnnInvoke.ReadNetFromDarknet(Pathconfig, PathWeights);
                    ClassLabels = File.ReadAllLines(PathClasses).ToList();
                    MessageBox.Show("MODEL LOADED SUCCESSFULL");
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }

        }
        private void Form1_Load(object sender, EventArgs e)
        {

        }
        private void imageToolStripMenuItem_Click(object sender, EventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog();

            int Width = 640;
            int Height = 480;
            dialog.Filter = "Image Files(*.jpg;*.png;)|*.jpg;*.png;";
            if (dialog.ShowDialog() == DialogResult.OK)
            {
                img = new Image<Bgr, byte>(dialog.FileName)
                        .Resize(Width, Height, Inter.Cubic);
                //img = ImageProcesse(img_);
                pictureBox1.Image = img.AsBitmap();
                Thread.Sleep(1000);
                MessageBox.Show("IMAGE SUCCESSFULL");
                Temp = 1;
            }
        }
        private void openToolStripMenuItem_Click(object sender, EventArgs e)
        {

            
        }
        private void loadModelToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {

                if (img == null)
                {
                    throw new Exception("Load Image!!!");
                }
                if (Model == null)
                {
                    throw new Exception("Load the model!!!");
                }
                float confThreshold = 0.8f;
                float nmsThreshold = 0.4f;


                var input = DnnInvoke.BlobFromImage(img, 1 / 255.0, swapRB: true);
                Model.SetInput(input);
                Model.SetPreferableBackend(Emgu.CV.Dnn.Backend.OpenCV);
                Model.SetPreferableTarget(Target.Cpu);

                VectorOfMat vectorOfMat = new VectorOfMat();
                Model.Forward(vectorOfMat, Model.UnconnectedOutLayersNames);

                // post processing
                VectorOfRect bboxes = new VectorOfRect();
                VectorOfFloat scores = new VectorOfFloat();
                VectorOfInt indices = new VectorOfInt();

                for (int k = 0; k < vectorOfMat.Size; k++)
                {
                    var mat = vectorOfMat[k];
                    var data = HelperClass.ArrayTo2DList(mat.GetData());

                    for (int i = 0; i < data.Count; i++)
                    {
                        var row = data[i];
                        var rowsscores = row.Skip(5).ToArray();
                        var classId = rowsscores.ToList().IndexOf(rowsscores.Max());
                        var confidence = rowsscores[classId];

                        if (confidence > confThreshold)
                        {
                            var center_x = (int)(row[0] * img.Width);
                            var center_y = (int)(row[1] * img.Height);

                            var width = (int)(row[2] * img.Width);
                            var height = (int)(row[3] * img.Height);

                            var x = (int)(center_x - (width / 2));
                            var y = (int)(center_y - (height / 2));

                            bboxes.Push(new Rectangle[] { new Rectangle(x, y, width, height) });
                            indices.Push(new int[] { classId });
                            scores.Push(new float[] { confidence });
                        }
                    }
                }

                var idx = DnnInvoke.NMSBoxes(bboxes.ToArray(), scores.ToArray(), confThreshold, nmsThreshold);

                var imgOutput = img.Clone();

                for (int i = 0; i < idx.Length; i++)
                {
                    int index = idx[i];
                    var bbox = bboxes[index];
                    imgOutput.Draw(bbox, new Bgr(0, 255, 0), 2);
                    CvInvoke.PutText(imgOutput, ClassLabels[indices[index]], new Point(bbox.X, bbox.Y + 20),
                        FontFace.HersheySimplex, 1.0, new MCvScalar(0, 0, 255), 2);
                }

                pictureBox1.Image = imgOutput.AsBitmap();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {
        
        }

        private void exitToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (MessageBox.Show("Are you sure you want to close ? ", "System Message", MessageBoxButtons.YesNo) == DialogResult.Yes)
            {
                this.Close();
            }
        }

        private void cameraToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                frame = new Mat();
                capture = new VideoCapture(cameradevice);
                capture.ImageGrabbed += ProcessFrame;
                capture.Start();
                captureInProgress = true;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error open camera: {ex.Message}");
            }
        }

        private void START_Click(object sender, EventArgs e)
        {
            try
            {
                if (captureInProgress)
                {
                    STOP_Click(sender, e);
                }
                if (!captureInProgress)
                {
                    cameraToolStripMenuItem_Click(sender, e);
                }

            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error starting camera: {ex.Message}");
            }
        }
        private void ProcessFrame(object sender, EventArgs e)
        {
            try
            {
                capture.Retrieve(frame, 0);
                pictureBox1.Image = frame.ToBitmap();
            }
            
            catch (Exception ex)
            {
                MessageBox.Show($"Error processing frame: {ex.Message}");
            }
        }

        private void STOP_Click(object sender, EventArgs e)
        {
            if (capture !=null)
            {
                try
                {
                    capture.ImageGrabbed -= ProcessFrame;
                    capture.Stop();
                    capture.Dispose();
                    captureInProgress = false;
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Error stopping camera: {ex.Message}");
                }
            }
        }

        

        private void CLOSE_Click(object sender, EventArgs e)
        {
            if(capture != null)
            {
                capture.Stop();
                capture.Dispose();
                captureInProgress = false;
                capture = null;
                pictureBox1.Image = null;
            }
            else if (img != null)
            {
                pictureBox1.Image = null;
                img = null;
            }
            else if (frame != null)
            {
                pictureBox1.Image = null;
            }
            else
            {
                pictureBox1.Image = null;
            }    
        }

        private void Cap_Ture_Click(object sender, EventArgs e)
        {
            
            capture = new VideoCapture();
            Mat frame = capture.QueryFrame();
            // Gán ảnh vào img
            img = frame.ToImage<Bgr, byte>();
            // hiển thị 
            pictureBox1.Image = null;
            pictureBox1.Image = frame.ToBitmap();

            
            loadModelToolStripMenuItem_Click(sender, e);
         
        }
        private static OpenFileDialog openfile()
        {
            OpenFileDialog dialog = new OpenFileDialog();
            dialog.Filter = "Video Files (*.mp4;*.avi;)|*.mp4;*.avi;";
            return dialog;
        }
        private void videoToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                OpenFileDialog dialog;
                dialog = openfile();
                //waveOut = new WaveOut();

                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    capture = new VideoCapture(dialog.FileName);
                    if (capture != null)
                    {
                        Mat frame = new Mat();
                        capture.Read(frame);
                        pictureBox1.Image = frame.ToBitmap();
                        Application.Idle += ProcessVideo_;
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void ProcessVideo_(object sender, EventArgs e )
        {
            try
            {
                Mat frame = capture.QueryFrame();
                if (frame.IsEmpty)
                {
                    Application.Idle -= ProcessVideo;
                    return;
                }
                pictureBox1.Image = frame.ToBitmap();
            }
            catch
            {

            }
        }

        private void objectDetectionVideoToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                if (capture != null)
                {
                    Mat frame = new Mat();
                    capture.Read(frame);
                    pictureBox1.Image = frame.ToBitmap();

                    backgroundSubtractor = new BackgroundSubtractorMOG2();

                    //backgroundSubtractor = new Emgu.CV.BgSegm.BackgroundSubtractorMOG();
                    Application.Idle += ProcessVideo;
                }


            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }
        private void ProcessVideo(object sender, EventArgs e)
        {
            try
            {
                Mat frame = capture.QueryFrame();
                if (frame.IsEmpty)
                {
                    Application.Idle -= ProcessVideo;
                    return;
                }

                Mat smoothFrame = new Mat();
                CvInvoke.GaussianBlur(frame, smoothFrame, new Size(3, 3), 1);

                Mat foregroundMask = new Mat();
                backgroundSubtractor.Apply(smoothFrame, foregroundMask);


                CvInvoke.Threshold(foregroundMask, foregroundMask, 200, 240, ThresholdType.Binary);
                CvInvoke.MorphologyEx(foregroundMask, foregroundMask, MorphOp.Close,
                    Mat.Ones(7, 3, DepthType.Cv8U, 1), new Point(-1, -1), 1, BorderType.Reflect, new MCvScalar(0));

                int minArea = 500;
                VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
                CvInvoke.FindContours(foregroundMask, contours, null, RetrType.External, ChainApproxMethod.ChainApproxSimple);
                for (int i = 0; i < contours.Size; i++)
                {
                    var bbox = CvInvoke.BoundingRectangle(contours[i]);
                    var area = bbox.Width * bbox.Height;
                    var ar = (float)bbox.Width / bbox.Height;

                    if (area > minArea && ar < 1.0)
                    {
                        CvInvoke.Rectangle(frame, bbox, new MCvScalar(0, 0, 255), 2);
                    }

                }

                pictureBox1.Image = frame.ToBitmap();
            }
            catch (Exception ex)
            {
                //throw new Exception(ex.Message);
            }


        }

        private void objectDetectionCameraToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                capture = new VideoCapture();
                if (capture != null)
                {
                    Mat frame = new Mat();
                    capture.Read(frame);
                    pictureBox1.Image = frame.ToBitmap();

                    //backgroundSubtractor = new BackgroundSubtractorMOG2();

                    backgroundSubtractor = new Emgu.CV.BgSegm.BackgroundSubtractorMOG();
                    Application.Idle += ProcessVideo;
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void groupBox1_Enter(object sender, EventArgs e)
        {

        }

        private void tableLayoutPanel3_Paint(object sender, PaintEventArgs e)
        {

        }

        private void button4_Click(object sender, EventArgs e)
        {
            readModelToolStripMenuItem_Click(sender, e);
        }

        private void tableLayoutPanel4_Paint(object sender, PaintEventArgs e)
        {

        }

        private void pictureBox2_Click(object sender, EventArgs e)
        {

        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {
            imageToolStripMenuItem_Click(sender, e);
        }

        private void button2_Click(object sender, EventArgs e)
        {
            videoToolStripMenuItem_Click(sender, e);
        }

        private void button3_Click(object sender, EventArgs e)
        {
            cameraToolStripMenuItem_Click(sender, e);
        }

        private void button5_Click(object sender, EventArgs e)
        {
            loadModelToolStripMenuItem_Click(sender, e);
        }

        private void button6_Click(object sender, EventArgs e)
        {
            objectDetectionVideoToolStripMenuItem_Click(sender, e);
        }

        private void button7_Click(object sender, EventArgs e)
        {
            objectDetectionCameraToolStripMenuItem_Click(sender, e);
        }

        private void button8_Click(object sender, EventArgs e)
        {
            exitToolStripMenuItem_Click(sender, e);
        }
    }
}
