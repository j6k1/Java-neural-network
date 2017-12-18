package j6k1.ai.nn;

public class NN {
	private final NNUnit[] units;
	private final double[][][] layers;

	public NN(NNUnit[] units, IInputReader reader, IInitialDataCreator initialDataCreator)
	{
		if(units.length < 3)
		{
			throw new InvalidConfigurationException(
					"Parameter of layer number of multilayer perceptron is incorrect (less than 3)");
		}
		else if(!(units[0].f instanceof FIdentity))
		{
			throw new InvalidConfigurationException(
					"Activation functions other than FIdentity can not be specified for the input layer.");
		}
		this.units = units;

		if(reader.sourceExists())
		{
			double[][][] layers = new double[units.length - 1][][];

			try {
				for(int i=0; i < units.length - 1; i++)
				{
					layers[i] = reader.readVec(units[i].size, units[i+1].size);
				}
			} catch (Exception e) {
				throw new RuntimeException(e);
			} finally {
				try {
					reader.close();
				} catch (Exception e) {
					throw new RuntimeException(e);
				}
			}
			this.layers = layers;
		}
		else
		{
			this.layers = initialDataCreator.create();
		}

		if(this.layers.length != units.length - 1)
		{
			throw new InvalidConfigurationException(
					"The layers count do not match. (units = " + units.length + ", layers = " + this.layers.length + ")");
		}
		else
		{
			for(int i=0, il=this.layers.length; i < il; i++)
			{
				if(units[i].size != this.layers[i].length)
				{
					throw new InvalidConfigurationException(
							"The units count do not match. (correct size = " + units[i].size + ", size = " + this.layers[i].length + ")");
				}

				for(int j=0, jl=units[i].size; j < jl; j++)
				{
					if(this.layers[i][j] == null)
					{
						throw new InvalidConfigurationException(
							String.format("Reference to unit %d of layer %d is null.",
								j, i
							));
					}
					else if(this.layers[i][j].length != units[i+1].size)
					{
						throw new InvalidConfigurationException(
							String.format(
								"Weight %d is defined for unit %d in layer %d, but this does not match the number of units in the lower layer.",
								this.layers[i][j].length, i, units[i+1].size
							)
						);
					}
				}
			}
		}
	}

	private NN(NNUnit[] units, double[][][] initialData)
	{
		this.units = units;
		this.layers = initialData;
	}

	private <T> T apply(int[] input, IAfterProcess<T> after)
	{
		if(input.length != units[0].size - 1)
		{
			throw new InvalidStateException(
				"The inputs to the input layer is invalid (the count of inputs must be the count of units -1)");
		}

		double[][] weighted = new double[units.length][];
		double[][] output = new double[units.length][];

		weighted[0] = new double[units[0].size];

		for(int k=1, kl=units[1].size; k < kl; k++)
		{
			weighted[0][k] += layers[0][0][k];
		}

		for(int j=1, jl=units[0].size; j < jl; j++)
		{
			for(int k=1, kl=units[1].size; k < kl; k++)
			{
				weighted[0][k] += input[j-1] * layers[0][j][k];
			}
		}

		output[0] = new double[units[0].size];

		for(int i=0, il=units.length - 1; i < il; i++)
		{
			weighted[i+1] = new double[units[i+1].size];
			IActivateFunction f = units[i].f;

			for(int j=0, jl = units[i].size; j < jl; j++)
			{
				output[i][j] = f.apply(weighted[i][j]);
			}

			output[i+1] = new double[units[i+1].size];

			for(int j=0, jl=units[i].size; j < jl; j++)
			{
				for(int k=1, kl = units[i+1].size; k < kl; k++)
				{
					output[i+1][k] += output[i][j] * layers[i][j][k];
					weighted[i+1][k] = output[i+1][k];
				}
			}
		}

		double[] result = new double[units[units.length-1].size];
		IActivateFunction f = units[units.length-1].f;

		for(int j=0, l=units.length, jl=units[units.length-2].size; j < jl; j++)
		{
			output[l-1] = new double[units[l-1].size];
			weighted[l-1] = new double[units[l-1].size];

			for(int k=0, kl = units[units.length-1].size; k < kl; k++)
			{
				output[l-1][k] += output[l-2][j] * layers[l-2][j][k];
				weighted[l-1][k] = output[l-1][k];
				result[k] = f.apply(weighted[l-1][k]);
			}
		}

		return after.apply(result, output, weighted);
	}

	public double[] solve(int[] input)
	{
		return apply(input, (result, output, weighted) -> new DoubleArray(result)).arr;
	}

	public NN learn(int[] input, double[] t)
	{
		if(t.length != units[units.length-1].size)
		{
			throw new InvalidStateException(
				"The number of answers input does not match the number of output units. You must enter data of the same count.");
		}

		return apply(input, (result, output, weighted) -> {

			double[][][] layers = new double[units.length-1][][];
			double[] delta = new double[units[units.length-1].size];

			for(int i=0, l = units.length-1; i < l; i++)
			{
				layers[i] = new double[units[i].size][];
			}

			IActivateFunction f = units[units.length-1].f;

			for(int j=0,
					l=units.length-2,
					kl=units[units.length-1].size,
					jl=units[units.length-2].size; j < jl; j++)
			{
				layers[l][j] = new double[kl];
			}

			for(int k=0,
					ul=units.length,
					kl=units[ul-1].size; k < kl; k++)
			{
				delta[k] = (result[k] - t[k]) * f.derive(weighted[ul-1][k]);

				for(int j=0, jl=units[units.length-2].size; j < jl; j++)
				{
					layers[ul-2][j][k] = this.layers[ul-2][j][k] - 0.5 * delta[k] * output[ul-2][j];
				}
			}

			for(int i=units.length - 2; i >= 1; i--)
			{
				double[] nextdelta = new double[units[i].size];

				for(int n=0, nl=units[i-1].size; n < nl; n++)
				{
					layers[i-1][n] = new double[units[i].size];
				}

				for(int j=0, jl=units[i].size; j < jl; j++)
				{
					layers[i-1][j] = new double[units[i-1].size];

					for(int k=0, kl=units[i+1].size; k < kl; k++)
					{
						nextdelta[j] += this.layers[i-1][j][k] * delta[k];
					}

					nextdelta[j] = nextdelta[j] * f.derive(weighted[i][j]);

					for(int n=0, nl=units[i-1].size; n < nl; n++)
					{
						layers[i-1][n][j] = this.layers[i-1][n][j] - 0.5 * nextdelta[j];
					}
				}

				delta = nextdelta;
			}

			return new NN(units, layers);
		});
	}

	public double[][][] getClonedLayers()
	{
		double[][][] layers = new double[this.layers.length][][];

		for(int i=0, il=this.layers.length; i < il; i++)
		{
			layers[i] = new double[this.layers[i].length][];

			for(int j=0, jl=this.layers[i].length; j < jl; j++)
			{
				layers[i][j] = new double[this.layers[i][j].length];

				for(int k=0, kl=this.layers[i][j].length; k < kl; k++)
				{
					layers[i][j][k] = this.layers[i][j][k];
				}
			}
		}

		return layers;
	}
}
class DoubleArray {
	public final double[] arr;

	public DoubleArray(double[] arr)
	{
		this.arr = arr;
	}
}
