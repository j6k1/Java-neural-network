package j6k1.ai.nn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;

public class TextFileInputReader implements IInputReader {
	private FileInputStream fin;
	private InputStreamReader in;
	private BufferedReader reader;
	private String[] line;
	private int index;
	private final boolean _sourceExists;
	public TextFileInputReader(File file) throws FileNotFoundException {
		_sourceExists = file.exists();

		if(_sourceExists)
		{
			fin = new FileInputStream(file);
			in = new InputStreamReader(fin);
			reader = new BufferedReader(in);
		}
	}

	@Override
	public boolean sourceExists() {
		return _sourceExists;
	}

	@Override
	public double[][] readVec(final int a, final int b) {
		double[][] result = new double[a][];

		for(int i=0; i < a; i++)
		{
			result[i] = new double[b];

			for(int j=0; j < b; j++)
			{
				try {
					result[i][j] = nextDouble();
				} catch (NumberFormatException | IOException e) {
					throw new RuntimeException(e);
				}
			}
		}

		return result;
	}

	public boolean close() {
		try {
			reader.close();
			in.close();
			fin.close();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		return true;
	}

	private String nextToken() throws IOException {
		if(line == null || index >= line.length)
		{
			String l;

			do {
				l = reader.readLine().trim();
			} while(l.isEmpty() || l.startsWith("#"));

			line = l.split(" ");
			index = 0;
		}

		return line[index++];
	}

	public double nextDouble() throws IOException, NumberFormatException {
		return Double.parseDouble(nextToken());
	}
}
