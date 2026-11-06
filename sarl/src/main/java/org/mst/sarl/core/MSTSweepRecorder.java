package org.mst.sarl.core;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Records one CSV row per role, scenario and threshold condition, checks the
 * verification contract, and audits nestedness between successive feasible
 * selections of the same scenario and role.
 *
 * <p>Verification contract: a feasible row is verified when it is sufficient,
 * privacy-admissible and deletion-minimal; an infeasible row is verified when it
 * is insufficient and selects no item. Nestedness is checked only when a
 * previous and a current feasible selection exist for the same scenario and
 * role. Because each threshold is optimized independently, nestedness is an
 * observed property of the run, not a guarantee.</p>
 */
public final class MSTSweepRecorder {

	public static final String FILE_NAME = "sarl_mst_multiscenario_sweep.csv";

	private final Path outputFile;
	private final Map<String, List<String>> previousFeasible = new HashMap<>();
	private PrintWriter writer;
	private int rows;
	private int verified;
	private int nestedApplicable;
	private int nestedPassed;

	public MSTSweepRecorder() {
		this(Paths.get("results", FILE_NAME));
	}

	public MSTSweepRecorder(Path outputFile) {
		this.outputFile = outputFile;
	}

	public synchronized void open() {
		try {
			Path parent = this.outputFile.toAbsolutePath().getParent();
			if (parent != null) {
				Files.createDirectories(parent);
			}
			this.writer = new PrintWriter(Files.newBufferedWriter(this.outputFile, StandardCharsets.UTF_8));
			this.writer.println("scenario,tau,role,status,selected_items,selected_count,"
					+ "utility,privacy_exposure,interpretation_cost,objective,sufficient,"
					+ "privacy_admissible,minimal,verified,"
					+ "nested_applicable,nested_passed");
			this.rows = 0;
			this.verified = 0;
			this.nestedApplicable = 0;
			this.nestedPassed = 0;
			this.previousFeasible.clear();
		} catch (IOException exception) {
			throw new IllegalStateException("Cannot open " + this.outputFile, exception);
		}
	}

	/** Records one response and returns its verification and nestedness status. */
	public synchronized RowStatus record(
			String scenario,
			double tau,
			String role,
			String status,
			List<String> selectedItems,
			double utility,
			double privacyExposure,
			double interpretationCost,
			double objective,
			boolean sufficient,
			boolean privacyAdmissible,
			boolean minimal) {
		boolean feasible = "FEASIBLE".equals(status);
		boolean rowVerified = feasible
				? sufficient && privacyAdmissible && minimal
				: !sufficient && selectedItems.isEmpty();
		if (rowVerified) {
			this.verified++;
		}

		boolean nestedApplicableForRow = false;
		boolean nestedPassedForRow = false;
		String key = scenario + "::" + role;
		if (feasible) {
			List<String> previous = this.previousFeasible.get(key);
			if (previous != null) {
				nestedApplicableForRow = true;
				this.nestedApplicable++;
				nestedPassedForRow = selectedItems.containsAll(previous);
				if (nestedPassedForRow) {
					this.nestedPassed++;
				}
			}
			this.previousFeasible.put(key, selectedItems);
		}

		writeRow(scenario, tau, role, status,
				selectedItems.isEmpty() ? "INFEASIBLE" : String.join("|", selectedItems),
				selectedItems.size(), utility, privacyExposure, interpretationCost, objective,
				sufficient, privacyAdmissible, minimal, rowVerified,
				nestedApplicableForRow, nestedPassedForRow);
		this.rows++;
		return new RowStatus(rowVerified, nestedApplicableForRow, nestedPassedForRow);
	}

	private void writeRow(
			String scenario, double tau, String role, String status, String items, int count,
			double utility, double privacyExposure, double interpretationCost, double objective,
			boolean sufficient,
			boolean privacyAdmissible,
			boolean minimal,
			boolean rowVerified,
			boolean nestedApplicable,
			boolean nestedPassedForRow) {
		this.writer.println(String.join(",",
				quote(scenario), Double.toString(tau), quote(role), quote(status), quote(items),
				Integer.toString(count),
				Double.toString(utility), Double.toString(privacyExposure),
				Double.toString(interpretationCost), Double.toString(objective),
				Boolean.toString(sufficient),
				Boolean.toString(privacyAdmissible),
				Boolean.toString(minimal),
				Boolean.toString(rowVerified),
				Boolean.toString(nestedApplicable),
				Boolean.toString(nestedPassedForRow)));
		this.writer.flush();
	}

	private static String quote(String value) {
		return "\"" + value + "\"";
	}

	public synchronized void close() {
		if (this.writer != null) {
			this.writer.close();
		}
	}

	public synchronized int getRows() {
		return this.rows;
	}

	public synchronized int getVerified() {
		return this.verified;
	}

	public synchronized int getNestedApplicable() {
		return this.nestedApplicable;
	}

	public synchronized int getNestedPassed() {
		return this.nestedPassed;
	}

	public Path getOutputFile() {
		return this.outputFile;
	}

	/** Verification and nestedness status of one recorded row. */
	public static final class RowStatus {
		private final boolean verified;
		private final boolean nestedApplicable;
		private final boolean nestedPassed;

		public RowStatus(
				boolean verified,
				boolean nestedApplicable,
				boolean nestedPassed) {
			this.verified = verified;
			this.nestedApplicable = nestedApplicable;
			this.nestedPassed = nestedPassed;
		}

		public boolean isVerified() {
			return this.verified;
		}

		public boolean isNestedApplicable() {
			return this.nestedApplicable;
		}

		public boolean isNestedPassed() {
			return this.nestedPassed;
		}
	}
}
