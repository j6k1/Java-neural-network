# Java-neural-network

## 使い方

```
NN nn = new NN(new NNUnit[] {
	new NNUnit(193),
	new NNUnit(193, new FReLU()),
	new NNUnit(193, new FReLU()),
	new NNUnit(4, new FTanh()),
	new NNUnit(1, new FTanh())
}, new TextFileInputReader(new File("data/nn.txt")), () -> {
	double[] initials = { 0.0001, 0.001 };
	int[] units = { 193, 193, 193 };
	double[][][] layers = new double[4][][];

	for(int i=0, l=units.length-1; i < l; i++)
	{
		layers[i] = new double[units[i]][];

		for(int j=0; j < units[i+1]; j++)
		{
			double[] weights = new double[units[i]];
			Arrays.fill(weights, initials[i]);
			layers[i][j] = weights;
		}
	}

	int runits[] = { 193, 4, 1 };

	Random r = new Random();

	for(int i=2, l=4; i < l; i++)
	{
		layers[i] = new double[runits[i-2]][];

		for(int j=0, ll=runits[i-2]; j < ll; j++)
		{
			layers[i][j] = new double[runits[i-1]];

			for(int k=0, lll=runits[i-1]; k < lll; k++)
			{
				layers[i][j][k] = (r.nextInt(100) + 1) * 0.1;
			}
		}
	}

	return layers;
});

Random r = new Random();

int[] input = new int[192];

for(int i=0; i < 192; i++)
{
	input[i] = r.nextInt(2);
}

long start = System.currentTimeMillis();

for(int i=0; i < 10; i++)
{
	double[] result = nn.solve(input);

	System.out.println(result[0]);
	nn = nn.learn(input, new double[] { 1.0 });
}

long end = System.currentTimeMillis();
```
