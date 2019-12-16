using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Windows.Forms;
using System.Collections;

namespace NeuralNetwork1
{
    /// <summary>
    /// Класс для хранения образа – входной массив сигналов на сенсорах, выходные сигналы сети, и прочее
    /// </summary>
    public class Sample
    {
        /// <summary>
        /// Входной вектор
        /// </summary>
        public double[] input = null;

        /// <summary>
        /// Выходной вектор, задаётся извне как результат распознавания
        /// </summary>
        public double[] output = null;

        /// <summary>
        /// Вектор ошибки, вычисляется по какой-нибудь хитрой формуле
        /// </summary>
        public double[] error = null;

        /// <summary>
        /// Действительный класс образа. Указывается учителем
        /// </summary>
        public FigureType actualClass;

        /// <summary>
        /// Распознанный класс - определяется после обработки
        /// </summary>
        public FigureType recognizedClass;

        /// <summary>
        /// Конструктор образа - на основе входных данных для сенсоров, при этом можно указать класс образа, или не указывать
        /// </summary>
        /// <param name="inputValues"></param>
        /// <param name="sampleClass"></param>
        public Sample(double[] inputValues, FigureType sampleClass = FigureType.Undef)
        {
            input = (double[])inputValues.Clone();
            recognizedClass = FigureType.Undef;
            actualClass = sampleClass;
        }

        /// <summary>
        /// Обработка реакции сети на данный образ на основе вектора выходов сети
        /// </summary>
        public void processOutput()
        {
            if (error == null)
                error = new double[output.Length];
            
            //  получаем ошибку и определённый класс
            recognizedClass = 0;
            for(int i = 0; i < output.Length; ++i)
            {
                error[i] = ((i == (int) actualClass ? 1 : 0) - output[i]);
                if (output[i] > output[(int)recognizedClass]) recognizedClass = (FigureType)i;
            }
        }

        /// <summary>
        /// Вычисленная суммарная квадратичная ошибка сети. Предполагается, что целевые выходы - 1 для верного, и 0 для остальных
        /// </summary>
        /// <returns></returns>
        public double EstimatedError()
        {
            double Result = 0;
            for (int i = 0; i < output.Length; ++i)
                Result += Math.Pow(error[i], 2);
            return Result;
        }

        /// <summary>
        /// Добавляет к аргументу ошибку, соответствующую данному образу (не квадратичную!!!)
        /// </summary>
        /// <param name="errorVector"></param>
        /// <returns></returns>
        public void updateErrorVector(double[] errorVector)
        {
            for (int i = 0; i < errorVector.Length; ++i)
                errorVector[i] += error[i];
        }

        /// <summary>
        /// Представление в виде строки
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            string result = "Sample decoding : " + actualClass.ToString() + "(" + ((int)actualClass).ToString() + "); " + Environment.NewLine + "Input : ";
            for (int i = 0; i < input.Length; ++i) result += input[i].ToString() + "; ";
            result += Environment.NewLine + "Output : ";
            if (output == null) result += "null;";
            else
                for (int i = 0; i < output.Length; ++i) result += output[i].ToString() + "; ";
            result += Environment.NewLine + "Error : ";

            if (error == null) result += "null;";
            else
                for (int i = 0; i < error.Length; ++i) result += error[i].ToString() + "; ";
            result += Environment.NewLine + "Recognized : " + recognizedClass.ToString() + "(" + ((int)recognizedClass).ToString() + "); " + Environment.NewLine;


            return result;
        }
        
        /// <summary>
        /// Правильно ли распознан образ
        /// </summary>
        /// <returns></returns>
        public bool Correct() { return actualClass == recognizedClass; }
    }
    
    /// <summary>
    /// Выборка образов. Могут быть как классифицированные (обучающая, тестовая выборки), так и не классифицированные (обработка)
    /// </summary>
    public class SamplesSet : IEnumerable
    {
        /// <summary>
        /// Накопленные обучающие образы
        /// </summary>
        public List<Sample> samples = new List<Sample>();
        
        /// <summary>
        /// Добавление образа к коллекции
        /// </summary>
        /// <param name="image"></param>
        public void AddSample(Sample image)
        {
            samples.Add(image);
        }
        public int Count { get { return samples.Count; } }

        public IEnumerator GetEnumerator()
        {
            return samples.GetEnumerator();
        }

        /// <summary>
        /// Реализация доступа по индексу
        /// </summary>
        /// <param name="i"></param>
        /// <returns></returns>
        public Sample this[int i]
        {
            get { return samples[i]; }
            set { samples[i] = value; }
        }

        public double ErrorsCount()
        {
            double correct = 0;
            double wrong = 0;
            foreach (var sample in samples)
                if (sample.Correct()) ++correct; else ++wrong;
            return correct / (correct + wrong);
        }
    }
    
    public class NeuralNetwork
    {
        /// <summary>
        /// Один нейрон сети
        /// </summary>
        private class Node
        {
            /// <summary>
            /// Входной взвешенный сигнал нейрона
            /// </summary>
            public double charge = 0;

            /// <summary>
            /// Выходной сигнал нейрона
            /// </summary>
            public double output = 0;
            
            /// <summary>
            /// Ошибка для данного нейрона
            /// </summary>
            public double error = 0;
            
            /// <summary>
            /// Сигнал поляризации (можно и 1 сделать в принципе)
            /// </summary>
            public static double biasSignal = -1.0;

            /// <summary>
            /// Генератор для инициализации весов
            /// </summary>
            private static Random randGenerator = new Random();
            
            /// <summary>
            /// Минимальное значение для начальной инициализации весов
            /// </summary>
            private static double initMinWeight = -1;
            
            /// <summary>
            /// Максимальное значение для начальной инициализации весов
            /// </summary>
            private static double initMaxWeight = 1;

            /// <summary>
            /// Количество узлов на предыдущем слое
            /// </summary>
            public int inputLayerSize = 0;

            /// <summary>
            /// Вектор входных весов нейрона
            /// </summary>
            public double[] weights = null;

            /// <summary>
            /// Вес на сигнале поляризации
            /// </summary>
            public double biasWeight = 0.01;

            /// <summary>
            /// Ссылка на предыдущий слой нейронов (здесь не создавать!!!)
            /// </summary>
            public Node[] inputLayer = null;

            /// <summary>
            /// Создаём один нейрон сети с ассоциированным вектором входящих весов. Предыдущий слой должен быть создан предварительно!!!!
            /// </summary>
            /// <param name="prevLayerNodes">Предыдущий слой как массив нейронов</param>
            public Node(Node[] prevLayerNodes)
            {
                inputLayer = prevLayerNodes;  //  Инициализируем ссылку на предыдущий слой. Если его нет - null

                if (prevLayerNodes == null) return;

                //  Если есть предыдущий слой
                inputLayerSize = prevLayerNodes.Length;
                //  Создаём вектор весов
                weights = new double[inputLayerSize];

                //  Инициализируем веса небольшими случайными значениями
                for (int i = 0; i < weights.Length; ++i)
                    weights[i] = initMinWeight + randGenerator.NextDouble() * (initMaxWeight - initMinWeight);
            }
            
            /// <summary>
            /// Просто вычисление пороговой функции (активация нейрона). Для сенсоров не вызывать!!!
            /// </summary>
            public void Activate()
            {
                // Перед вычислением функции активации добавляем взвешенное значение сигнала поляризации
                charge = biasWeight * biasSignal;
                for (int i = 0; i < inputLayer.Length; ++i)
                    charge += inputLayer[i].output * weights[i];
                //  Считаем выход нейрона
                output = ActivationFunction(charge);
                //  Если посчитан выход, то можно сбросить входной сигнал
                charge = 0;
            }
            
            
            /// <summary>
            /// Распространение ошибки на предыдущий слой и пересчёт весов. Предварительно там ошибка должна быть сброшена (хотя бы в процессе обновления весов)
            /// </summary>
            public void BackpropError(double ita)
            {
                //  Сначала обрабатываем ошибку собственно в текущем нейроне
                error *= output * (1 - output);

                //  Теперь разбираемся с сигналом поляризации - он имеет выход -1 и вес biasWeight, его пересчитываем
                biasWeight += ita * error * biasSignal;

                for (int i = 0; i < inputLayerSize; i++)
                    inputLayer[i].error += error * weights[i];
                //  ошибку пробросили на предыдущий слой, теперь меняем веса
                for (int i = 0; i < inputLayerSize; i++)
                    weights[i] += ita * error * inputLayer[i].output;
                //  сбрасываем ошибку
                error = 0;
            }
            
            /// <summary>
            /// Функция активации
            /// </summary>
            /// <param name="inp">Вход (взвешенная сумма входных сигналов)</param>
            /// <returns></returns>
            public static double ActivationFunction(double inp)
            {
                return 1 / (1 + Math.Exp(-inp));
            }
        }




       
        /// <summary>
        /// Скорость обучения. В процессе пока что не меняем, но мало ли
        /// </summary>
        public double LearningSpeed = 0.01;
        private Node[] Sensors;
        private Node[][] Layers;  //  В этом массиве действительно создаются нейрончики, остальные массивы - просто ссылки на первый и последний
        private Node[] Outputs;

        private void Init(int[] structure) {
            //  В структуре как минимум должны быть 2 слоя - сенсоры и выход. Кстати, иногда будет работать!
            if (structure.Length < 2)
                throw new Exception("Invalid net structure!");

            //  Массивик создаём
            Layers = new Node[structure.Length][];

            // Сенсоры отдельно создаём, у них входов нет
            Layers[0] = new Node[structure[0]];
            for (int neuron = 0; neuron < structure[0]; ++neuron)
                Layers[0][neuron] = new Node(null);
            Sensors = Layers[0];

            //  Остальные слои по порядку, указывая каждому нейрону в качестве входа предыдущий слой
            for (int layer = 1; layer < structure.Length; ++layer)
            {
                Layers[layer] = new Node[structure[layer]];   //  Выделили память под слой
                for (int neuron = 0; neuron < structure[layer]; ++neuron)
                    Layers[layer][neuron] = new Node(Layers[layer - 1]);  //  И по одному нейрону создаём. Каждый нейрон должен знать ссылку на предыдущий слой
            }
            //  Ссылку на выходной слой оставляем
            Outputs = Layers[Layers.Length - 1];
        }
        
        /// <summary>
        /// Конструктор нейросети – с массивом, определяющим структуру сети
        /// </summary>
        /// <param name="structure"></param>
        public NeuralNetwork(int[] structure)
        {
            Init(structure);
        }

        public NeuralNetwork(string path)
        {
            LoadFromFile(path);
        }

        /// <summary>
        /// Прямой однократный прогон сети
        /// </summary>
        /// <param name="image">Входной образ для обработки</param>
        private void Run(Sample image)
        {
            if (image.input.Length != Sensors.Length)
                throw new Exception("Ошибка");
            
            //  Перекидываем значения на сенсорный слой
            for (int i = 0; i < image.input.Length; i++)
                Sensors[i].output = image.input[i];

            //  По всем слоям кроме сенсорного выполняем обработку
            for (int i = 1; i < Layers.Length; i++)
                for (int j = 0; j < Layers[i].Length; j++)
                    Layers[i][j].Activate();

            // Закидываем обратно в образ результат обработки
            if (image.output == null) 
                image.output = new double[Layers[Layers.Length - 1].Length];
            
            for (int i = 0; i < Layers[Layers.Length - 1].Length; i++)
                image.output[i] = Layers[Layers.Length - 1][i].output;

            image.processOutput();  //  Тут и ошибка посчитается, и всё прочее
        }
        
        /// <summary>
        /// Обратно прогоняем ошибку, ну и пересчитываем веса
        /// </summary>
        /// <param name="image">Образ, содержащий ошибку выходного слоя</param>
        /// <param name="ita"></param>
        private void BackProp(Sample image, double ita)
        {
            //  Считываем ошибку из образа на выходной слой
            for (int i = 0; i < Layers[Layers.Length - 1].Length; i++)
                Layers[Layers.Length - 1][i].error = image.error[i];
            
            for (int i = Layers.Length - 1; i >= 0; --i)
                for (int j = 0; j < Layers[i].Length; ++j)
                    Layers[i][j].BackpropError(ita);
        }

        /// <summary>
        /// Распознавание одного образа
        /// </summary>
        /// <param name="sample">Входной образ</param>
        /// <returns>Класс фигуры</returns>
        public FigureType predict(Sample sample)
        {
            //  Прогоняем
            Run(sample);
            //  Возвращаем распознанный класс
            return sample.recognizedClass;
        }

        /// <summary>
        /// Обучение одному заданному образу
        /// </summary>
        /// <param name="sample">Образец для обучения</param>
        /// <returns>Число итераций, выполненных для обучения данному образу. Если 0 - сразу распознали верно</returns>
        public int Train(Sample sample)
        {
            int iters = 0;
            while(iters<100)
            {
                //  Прямой прогон сети
                Run(sample);
                if (sample.EstimatedError() < 0.2 && sample.Correct())
                {
                    Debug.WriteLine("Управились за " + iters.ToString());
                    return iters;
                }

                ++iters;
                // Один шаг обратного прогона и пересчёта весов
                BackProp(sample, LearningSpeed);
            }
            if (iters == 100) Debug.WriteLine("Опаньки, какой тяжелый образ!");
            return iters;
        }
        
        /// <summary>
        /// Вектор выходных значений
        /// </summary>
        /// <returns></returns>
        public double[] getOutput()
        {
            return Outputs.Select(n => n.output).ToArray();
        }

        /// <summary>
        /// Обучаем сеть на заданном датасете
        /// </summary>
        /// <param name="samplesSet">Обучающая выборка</param>
        /// <param name="epochs_count">Количество проходов по обучающей выборке</param>
        /// <param name="acceptable_erorr">Допустимая ошибка</param>
        /// <returns>Процент верно распознанных образов на последней итерации</returns>
        public double TrainOnDataSet(SamplesSet samplesSet, int epochs_count, double acceptable_erorr)
        {
            double guessLevel = 0;
            do
            {
                guessLevel = 0;
                for (int i = 0; i < samplesSet.samples.Count; ++i)
                    if (Train(samplesSet.samples.ElementAt(i)) == 0)
                        guessLevel += 1;
                //  Тут просто процент верно распознанных образов
                guessLevel /= samplesSet.samples.Count;
                if (guessLevel > acceptable_erorr) return guessLevel;
                epochs_count--;
            } while (epochs_count > 0);

            // Возвращаем результат
            return guessLevel;
        }

        public double TestOnDataSet(SamplesSet testSet)
        {
            if (testSet.Count == 0) return double.NaN;

            double guessLevel = 0;
            for (int i = 0; i < testSet.Count; ++i)
            {
                Sample s = testSet.samples.ElementAt(i);
                predict(s);
                if (s.Correct()) guessLevel += 1;
            }
            return guessLevel / testSet.Count;
        }


        public void SaveToFile(string Path)
        {
            StringBuilder builder = new StringBuilder();
            builder.AppendLine(String.Join("\t", Layers.Select(l => l.GetLength(0))));
            foreach(var layer in Layers)
            {
                foreach(var node in layer)
                {
                    if(node.weights == null)
                    {
                        builder.AppendLine((0.0).ToString());
                    }
                    else 
                     builder.AppendLine(String.Join("\t", node.weights.Select(w => w.ToString()))); 
                }
            }
            System.IO.File.WriteAllText(Path, builder.ToString());
        }
        public void LoadFromFile(string Path)
        {
            string[] lines = System.IO.File.ReadAllLines(Path);
            int [] structure = lines[0].Split(new char[] { '\t' }, StringSplitOptions.RemoveEmptyEntries).Select(s => int.Parse(s)).ToArray();

            Init(structure);
            int index = 1;
            foreach (var layer in Layers)
            {
                foreach (var node in layer)
                {
                    node.weights = lines[index].Split(new char[] { '\t' }, StringSplitOptions.RemoveEmptyEntries).Select(s => Double.Parse(s)).ToArray();
                    index++;
                }
            }


        }

    }
}
