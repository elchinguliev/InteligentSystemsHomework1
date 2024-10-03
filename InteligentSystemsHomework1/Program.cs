using System;
using System.Collections.Generic;
using System.Linq;

class KMeans
{
    private int k;
    private List<double[]> centroids;
    private List<double[]> dataPoints;

    public KMeans(int k)
    {
        this.k = k;
        centroids = new List<double[]>();
        dataPoints = new List<double[]>();
    }

    public void Fit(List<double[]> data)
    {
        dataPoints = data;
        InitializeCentroids();

        bool centroidsChanged;
        int iterations = 0;

        do
        {
            centroidsChanged = false;
            List<List<double[]>> clusters = CreateClusters();

            List<double[]> newCentroids = new List<double[]>();
            foreach (var cluster in clusters)
            {
                if (cluster.Count > 0)
                {
                    double[] newCentroid = CalculateMean(cluster);
                    newCentroids.Add(newCentroid);
                }
            }

            if (!AreCentroidsSame(newCentroids))
            {
                centroids = newCentroids;
                centroidsChanged = true;
            }

            iterations++;
        } while (centroidsChanged && iterations < 100);

        Console.WriteLine("Clustering completed in {0} iterations.", iterations);
    }

    public void Predict(List<double[]> newData)
    {
        foreach (var point in newData)
        {
            int clusterIndex = GetNearestCentroidIndex(point);
            Console.WriteLine("Point {0},{1} is in cluster {2}.", point[0], point[1], clusterIndex + 1);
        }
    }

    private void InitializeCentroids()
    {
        Random random = new Random();
        for (int i = 0; i < k; i++)
        {
            centroids.Add(dataPoints[random.Next(dataPoints.Count)]);
        }
    }

    private List<List<double[]>> CreateClusters()
    {
        List<List<double[]>> clusters = new List<List<double[]>>(k);
        for (int i = 0; i < k; i++)
        {
            clusters.Add(new List<double[]>());
        }

        foreach (var point in dataPoints)
        {
            int nearestCentroidIndex = GetNearestCentroidIndex(point);
            clusters[nearestCentroidIndex].Add(point);
        }

        return clusters;
    }

    private int GetNearestCentroidIndex(double[] point)
    {
        double minDistance = double.MaxValue;
        int index = -1;

        for (int i = 0; i < centroids.Count; i++)
        {
            double distance = EuclideanDistance(point, centroids[i]);
            if (distance < minDistance)
            {
                minDistance = distance;
                index = i;
            }
        }

        return index;
    }

    private double EuclideanDistance(double[] point1, double[] point2)
    {
        return Math.Sqrt(Math.Pow(point1[0] - point2[0], 2) + Math.Pow(point1[1] - point2[1], 2));
    }

    private double[] CalculateMean(List<double[]> cluster)
    {
        double[] mean = new double[2];
        foreach (var point in cluster)
        {
            mean[0] += point[0];
            mean[1] += point[1];
        }
        mean[0] /= cluster.Count;
        mean[1] /= cluster.Count;
        return mean;
    }

    private bool AreCentroidsSame(List<double[]> newCentroids)
    {
        for (int i = 0; i < centroids.Count; i++)
        {
            if (EuclideanDistance(centroids[i], newCentroids[i]) > 1e-4)
            {
                return false;
            }
        }
        return true;
    }
}

class Program
{
    static void Main(string[] args)
    {
        List<double[]> data = new List<double[]>
        {
            new double[] { 1.0, 2.0 },
            new double[] { 1.5, 1.8 },
            new double[] { 5.0, 8.0 },
            new double[] { 8.0, 8.0 },
            new double[] { 1.0, 0.6 },
            new double[] { 9.0, 11.0 }
        };

        //EXAMPLE 2 FOR ADDITIONAL POINT
        //List<double[]> data = new List<double[]>
        //{
        //    new double[] { 1.0, 2.0 },
        //    new double[] { 2.5, 3.8 },
        //    new double[] { 6.0, 7.5 },
        //    new double[] { 8.5, 9.0 },
        //    new double[] { 10.0, 10.5 },
        //    new double[] { 1.5, 2.5 }
        //};


        KMeans kmeans = new KMeans(2);
        kmeans.Fit(data);

        List<double[]> testData = new List<double[]>
        {
            new double[] { 2.0, 1.0 },
            new double[] { 6.0, 9.0 }
        };

        kmeans.Predict(testData);
    }
}
