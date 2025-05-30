using System;
using System.Linq;

class XORGenetyczny
{
    static Random random = new Random();

    static void Main()
    {
        double ZDMin = -10, ZDMax = 10;
        int LBnP = 4;
        int liczbaWag = 9;
        int LBnOs = LBnP * liczbaWag;
        int liczbaOsobnikow = 13;
        int liczbaIteracji = 100;
        int rozmiarTurnieju = 3;
        double prawdopodobienstwoMutacji = 0.1;

        double[][] wejscia = new double[][]
        {
            new double[] { 0, 0, 0 },
            new double[] { 1, 0, 1 },
            new double[] { 0, 1, 0 },
            new double[] { 1, 1, 0 }
        };

        double[] wyjscia = new double[] { 0, 1, 1, 0 };

        int[][] populacja = new int[liczbaOsobnikow][];
        for (int i = 0; i < liczbaOsobnikow; i++)
        {
            populacja[i] = new int[LBnOs];
            for (int j = 0; j < LBnOs; j++)
            {
                populacja[i][j] = random.Next(2);
            }
        }

        double najlepszyBladGlobalny = double.MaxValue;
        int[] najlepszyOsobnik = null;

        for (int iter = 0; iter < liczbaIteracji; iter++)
        {
            double[] oceny = new double[liczbaOsobnikow];

            for (int i = 0; i < liczbaOsobnikow; i++)
            {
                double[] wagi = new double[liczbaWag];
                for (int w = 0; w < liczbaWag; w++)
                {
                    int[] gen = populacja[i].Skip(w * LBnP).Take(LBnP).ToArray();
                    wagi[w] = Dekoduj(gen, ZDMin, ZDMax, LBnP);
                }

                double blad = 0;
                for (int p = 0; p < wejscia.Length; p++)
                {
                    double[] ukryte = new double[2];
                    ukryte[0] = Sigmoid(wagi[0] * wejscia[p][0] + wagi[1] * wejscia[p][1] + wagi[2] * wejscia[p][2]);
                    ukryte[1] = Sigmoid(wagi[3] * wejscia[p][0] + wagi[4] * wejscia[p][1] + wagi[5] * wejscia[p][2]);

                    double wyj = Sigmoid(wagi[6] * ukryte[0] + wagi[7] * ukryte[1] + wagi[8] * 1);

                    blad += Math.Pow(wyjscia[p] - wyj, 2);
                }
                oceny[i] = blad;

                if (blad < najlepszyBladGlobalny)
                {
                    najlepszyBladGlobalny = blad;
                    najlepszyOsobnik = populacja[i].ToArray();
                }
            }

            int[][] nowaPopulacja = new int[liczbaOsobnikow][];
            for (int i = 0; i < liczbaOsobnikow / 2; i++)
            {
                int r1 = SelekcjaTurniejowa(Enumerable.Range(0, liczbaOsobnikow).ToArray(), oceny, rozmiarTurnieju);
                int r2 = SelekcjaTurniejowa(Enumerable.Range(0, liczbaOsobnikow).ToArray(), oceny, rozmiarTurnieju);
                (int[] d1, int[] d2) = OperatorKrzyzowania(populacja[r1], populacja[r2], LBnOs);
                nowaPopulacja[i * 2] = OperatorMutacji(d1, LBnOs, prawdopodobienstwoMutacji);
                nowaPopulacja[i * 2 + 1] = OperatorMutacji(d2, LBnOs, prawdopodobienstwoMutacji);
            }

            int najlepszyIndexIteracji = Array.IndexOf(oceny, oceny.Min());
            nowaPopulacja[liczbaOsobnikow - 1] = populacja[najlepszyIndexIteracji].ToArray();

            double najlepszyBladIteracji = oceny.Min();
            double sredniBlad = oceny.Average();
            Console.WriteLine($"Iteracja {iter + 1}: Najlepszy błąd = {najlepszyBladIteracji:F6}, Średni błąd = {sredniBlad:F6}");

            populacja = nowaPopulacja;
        }

        Console.WriteLine($"\nNajlepszy osiągnięty błąd: {najlepszyBladGlobalny:F6}");

        double[] najlepszeWagi = new double[liczbaWag];
        for (int w = 0; w < liczbaWag; w++)
        {
            int[] gen = najlepszyOsobnik.Skip(w * LBnP).Take(LBnP).ToArray();
            najlepszeWagi[w] = Dekoduj(gen, ZDMin, ZDMax, LBnP);
        }

        Console.WriteLine("\nNajlepsze wagi:");
        for (int w = 0; w < liczbaWag; w++)
        {
            Console.WriteLine($"w{w}: {najlepszeWagi[w]:F4}");
        }

        Console.WriteLine("\nTestowanie najlepszego rozwiązania:");
        for (int p = 0; p < wejscia.Length; p++)
        {
            double[] ukryte = new double[2];
            ukryte[0] = Sigmoid(najlepszeWagi[0] * wejscia[p][0] + najlepszeWagi[1] * wejscia[p][1] + najlepszeWagi[2] * wejscia[p][2]);
            ukryte[1] = Sigmoid(najlepszeWagi[3] * wejscia[p][0] + najlepszeWagi[4] * wejscia[p][1] + najlepszeWagi[5] * wejscia[p][2]);
            double wyj = Sigmoid(najlepszeWagi[6] * ukryte[0] + najlepszeWagi[7] * ukryte[1] + najlepszeWagi[8] * 1);
            Console.WriteLine($"Wejście: {string.Join(", ", wejscia[p])} => Wyjście: {wyj:F4} (Oczekiwane: {wyjscia[p]})");
        }
    }

    public static double Sigmoid(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }

    public static double Dekoduj(int[] cb, double ZDMin, double ZDMax, int LBnP)
    {
        double ZD = ZDMax - ZDMin;
        int ctmp = 0;
        for (int b = 0; b < LBnP; b++)
        {
            ctmp += cb[LBnP - 1 - b] * (int)Math.Pow(2, b);
        }
        return ZDMin + (ctmp / (Math.Pow(2, LBnP) - 1)) * ZD;
    }

    public static int[] OperatorMutacji(int[] cbwe, int LBnOs, double prawdopodobienstwo)
    {
        int[] cbwy = new int[LBnOs];
        for (int i = 0; i < LBnOs; i++)
        {
            if (random.NextDouble() < prawdopodobienstwo)
                cbwy[i] = cbwe[i] == 0 ? 1 : 0;
            else
                cbwy[i] = cbwe[i];
        }
        return cbwy;
    }

    public static (int[], int[]) OperatorKrzyzowania(int[] cbr1, int[] cbr2, int LBnOs)
    {
        int ciecie = random.Next(0, LBnOs);
        int[] cbp1 = new int[LBnOs];
        int[] cbp2 = new int[LBnOs];
        for (int b = 0; b <= ciecie; b++)
        {
            cbp1[b] = cbr1[b];
            cbp2[b] = cbr2[b];
        }
        for (int b = ciecie + 1; b < LBnOs; b++)
        {
            cbp1[b] = cbr2[b];
            cbp2[b] = cbr1[b];
        }
        return (cbp1, cbp2);
    }

    public static int SelekcjaTurniejowa(int[] Pulaosobnikow, double[] Ocenaosobnikow, int TurRozm)
    {
        int[] SkladTurnieju = Pulaosobnikow.OrderBy(x => random.Next()).Take(TurRozm).ToArray();
        return SkladTurnieju.OrderBy(i => Ocenaosobnikow[i]).First();
    }
}