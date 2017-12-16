package j6k1.ai.nn;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Arrays;

public class PersistenceWithTextFile implements IPersistence {
	private final File dst;

	public PersistenceWithTextFile(File file)
	{
		dst = file;
	}
	@Override
	public boolean save(NN nn) {
		try(FileOutputStream ostream = new FileOutputStream(dst, false);
				OutputStreamWriter swriter = new OutputStreamWriter(ostream, "UTF-8");
				BufferedWriter writer = new BufferedWriter(swriter)) {
			double[][][] layers = nn.getClonedLayers();

			writer.write("#Java NN config start.");

			for(int i=0, il=layers.length; i < il; i++)
			{
				writer.write("#layer: " + (i - 1));

				for(int j=0, jl=layers[i].length; j < jl; j++)
				{
					writer.write(
						String.join(" ", Arrays
											.stream(layers[i][j])
											.mapToObj(w -> Double.toString(w)).toArray(String[]::new)));
					writer.write('\n');
				}
			}

			writer.write("#endofile.");
			writer.flush();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

		return true;
	}
}
