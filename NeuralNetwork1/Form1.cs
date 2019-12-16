using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NeuralNetwork1
{
    public partial class Form1 : Form
    {
        GenerateImage generator = new GenerateImage();
        NeuralNetwork net = null;

        public Form1()
        {
            InitializeComponent();
            generator.figure_count = (int)classCounter.Value;
            button3_Click(this, null);
            pictureBox1.Image = Properties.Resources.Title;
        }

        private void set_result(Sample figure)
        {
            label1.Text = figure.ToString();

            if (figure.Correct())
                label1.ForeColor = Color.Green;
            else
                label1.ForeColor = Color.Red;

            label1.Text = "Распознано : " + figure.recognizedClass.ToString();

            label8.Text = String.Join("\n", net.getOutput().Select(d => d.ToString()));
            pictureBox1.Image = generator.genBitmap();
            pictureBox1.Invalidate();
        }

        private void pictureBox1_MouseClick(object sender, MouseEventArgs e)
        {
            Sample fig = generator.GenerateFigure();
            Enabled = false;
            net.predict(fig);
            Enabled = true;
            set_result(fig);
        }

        private async Task<double> train_networkAsync(int training_size, int epoches, double acceptable_error)
        {
            //  Выключаем всё ненужное
            label1.Text = "Выполняется обучение...";
            label1.ForeColor = Color.Red;
            groupBox1.Enabled = false;
            pictureBox1.Enabled = false;
            btnTrainOne.Enabled = false;

            //  Создаём новую обучающую выборку
            SamplesSet samples = new SamplesSet();

            for (int i = 0; i < training_size; i++)
                samples.AddSample(generator.GenerateFigure());

            //  Обучение запускаем асинхронно, чтобы не блокировать форму
            double f = await Task.Run(() => net.TrainOnDataSet(samples, epoches, acceptable_error));

            label1.Text = "Щелкните на картинку для теста нового образа";
            label1.ForeColor = Color.Green;
            groupBox1.Enabled = true;
            pictureBox1.Enabled = true;
            btnTrainOne.Enabled = true;
            StatusLabel.Text = "Accuracy: " + f.ToString();
            StatusLabel.ForeColor = Color.Green;
            return f;

        }

        private void button1_Click(object sender, EventArgs e)
        {
            #pragma warning disable CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed
            train_networkAsync( (int)TrainingSizeCounter.Value, (int)EpochesCounter.Value, (100 - AccuracyCounter.Value) / 100.0);
            #pragma warning restore CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed
        }

        private void button2_Click(object sender, EventArgs e)
        {
            this.Enabled = false;
            //  Тут просто тестирование новой выборки
            //  Создаём новую обучающую выборку
            SamplesSet samples = new SamplesSet();

            for (int i = 0; i < (int)TrainingSizeCounter.Value; i++)
                samples.AddSample(generator.GenerateFigure());

            double accuracy = net.TestOnDataSet(samples);
            
            StatusLabel.Text = string.Format("Точность на тестовой выборке : {0,5:F2}%", accuracy*100);
            if (accuracy*100 >= AccuracyCounter.Value)
                StatusLabel.ForeColor = Color.Green;
            else
                StatusLabel.ForeColor = Color.Red;

            this.Enabled = true;
        }

        private void button3_Click(object sender, EventArgs e)
        {
            //  Проверяем корректность задания структуры сети
            int[] structure = netStructureBox.Text.Split(';').Select((c) => int.Parse(c)).ToArray();
            if (structure.Length < 2 || structure[0] != 400 || structure[structure.Length - 1] != generator.figure_count)
            {
                MessageBox.Show("А давайте вы структуру сети нормально запишите, ОК?", "Ошибка", MessageBoxButtons.OK);
                return;
            };

            net = new NeuralNetwork(structure);
        }

        private void classCounter_ValueChanged(object sender, EventArgs e)
        {
            generator.figure_count = (int)classCounter.Value;
        }

        private void btnTrainOne_Click(object sender, EventArgs e)
        {
            if (net == null) return;
            Sample fig = generator.GenerateFigure();
            pictureBox1.Image = generator.genBitmap();
            pictureBox1.Invalidate();
            net.Train(fig);
            set_result(fig);
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void button4_Click(object sender, EventArgs e)
        {
            var r =  saveFileDialog1.ShowDialog();
            switch (r)
            {
                
                case DialogResult.OK:
                case DialogResult.Yes:
                    net.SaveToFile(saveFileDialog1.FileName);
                    StatusLabel.Text = "Сеть сохранена";
                    break;
                default:
                    break;
            }
        }

        private void button5_Click(object sender, EventArgs e)
        {
            var r = openFileDialog1.ShowDialog();
            switch (r)
            {

                case DialogResult.OK:
                case DialogResult.Yes:
                    try
                    {
                        var loadedNet = new NeuralNetwork(openFileDialog1.FileName);
                        net = loadedNet;
                        StatusLabel.Text = "Сеть загружена";
                    }
                    catch (Exception)
                   {
                        StatusLabel.Text = "Ошибка загрузки сети";
                        System.Media.SystemSounds.Hand.Play();

                    }

                    break;
                default:
                    break;
            }
        }
    }

  }
