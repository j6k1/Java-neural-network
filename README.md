# Java-neural-network

## 使い方

```java:sample
final int width = 4;
final int depth = 4;
NN nn = new NN(new NNUnit[] {
		new NNUnit(2),
		new NNUnit(width, new FReLU()),
		new NNUnit(width, new FReLU()),
		new NNUnit(1, new FTanh()),
	}, new TextFileInputReader(new File("data/nn.txt")), () -> {
		double[][][] layers = new double[depth-1][][];

		Random r = new Random();

		layers[0] = new double[3][];

		layers[0][0] = new double[width];

		for(int k=0; k < width; k++)
		{
			layers[0][0][k] = 0.0;
		}

		for(int j=1; j < 3; j++)
		{
			layers[0][j] = new double[width];

			for(int k=0; k < width; k++)
			{
				layers[0][j][k] = r.nextDouble() * (r.nextInt(2) == 1 ? 1.0 : -1.0);
			}
		}

		for(int i=1; i < depth-2; i++)
		{
			layers[i] = new double[width+1][];

			layers[i][0] = new double[width];

			for(int k=0; k < width; k++)
			{
				layers[i][0][k] = 0.0;
			}

			for(int j=1; j < width+1; j++)
			{
				layers[i][j] = new double[width];

				for(int k=0; k < width; k++)
				{
					layers[i][j][k] = r.nextDouble() * (r.nextInt(2) == 1 ? 1.0 : -1.0);
				}
			}
		}

		layers[depth-2] = new double[width+1][];

		layers[depth-2][0] = new double[1];

		for(int k=0; k < 1; k++)
		{
			layers[depth-2][0][k] = 0.0;
		}

		for(int j=1; j < width+1; j++)
		{
			layers[depth-2][j] = new double[1];

			for(int k=0; k < 1; k++)
			{
				layers[depth-2][j][k] = r.nextDouble() * (r.nextInt(2) == 1 ? 1.0 : -1.0);
			}
		}

		return layers;
	});

Pair[] pairs = new Pair[] {
	new Pair(new int[] { 0, 0 }, new int[] { 0 }),
	new Pair(new int[] { 0, 1 }, new int[] { 1 }),
	new Pair(new int[] { 1, 0 }, new int[] { 1 }),
	new Pair(new int[] { 1, 1 }, new int[] { 0 }),
};

(new PersistenceWithTextFile(new File("data/nn.initial.txt"))).save(nn);

Random r = new Random();

for(int i=0; i < 10000; i++)
{
	//int ii = r.nextInt(4);
	for(int ii=0; ii < 4; ii++)
	{
		double[] nnanswer = nn.solve(pairs[ii].input);

		nn = nn.learn(pairs[ii].input,
						Arrays.stream(pairs[ii].answer)
							.mapToDouble(m -> (double)m).toArray(), 0.5);
	}
}

for(Pair p: pairs)
{
	System.out.println("correct answer = " + String.join(", ", Arrays.stream(p.answer)
												.mapToObj(m -> Integer.toString(m))
												.toArray(String[]::new)));
	double[] nnanswer = nn.solve(p.input);
	System.out.println("nn answer = " +
			String.join(", ", Arrays.stream(nnanswer)
					.mapToObj(m -> Double.toString(m))
					.toArray(String[]::new)));
}
```
