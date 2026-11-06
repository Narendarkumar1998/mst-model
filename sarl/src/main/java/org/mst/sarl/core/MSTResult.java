package org.mst.sarl.core;

import java.util.Collections;
import java.util.List;

/**
 * Immutable result of one Exact-MST evaluation for a role, scenario and threshold.
 */
public final class MSTResult {

	private final String status;
	private final List<String> selectedItems;
	private final double utility;
	private final double privacyExposure;
	private final double interpretationCost;
	private final double objective;
	private final boolean sufficient;
	private final boolean privacyAdmissible;
	private final boolean minimal;

	public MSTResult(
			String status,
			List<String> selectedItems,
			double utility,
			double privacyExposure,
			double interpretationCost,
			double objective,
			boolean sufficient,
			boolean privacyAdmissible,
			boolean minimal) {
		this.status = status;
		this.selectedItems = Collections.unmodifiableList(selectedItems);
		this.utility = utility;
		this.privacyExposure = privacyExposure;
		this.interpretationCost = interpretationCost;
		this.objective = objective;
		this.sufficient = sufficient;
		this.privacyAdmissible = privacyAdmissible;
		this.minimal = minimal;
	}

	public static MSTResult infeasible() {
		return new MSTResult("INFEASIBLE", List.of(), 0.0, 0.0, 0.0, 0.0, false, false, false);
	}

	public boolean isFeasible() {
		return "FEASIBLE".equals(this.status);
	}

	public String getStatus() {
		return this.status;
	}

	public List<String> getSelectedItems() {
		return this.selectedItems;
	}

	/** Selected identifiers joined by '|', or INFEASIBLE when nothing is selected. */
	public String getSelectedItemsAsString() {
		return this.selectedItems.isEmpty() ? "INFEASIBLE" : String.join("|", this.selectedItems);
	}

	public int getSelectedCount() {
		return this.selectedItems.size();
	}

	public double getUtility() {
		return this.utility;
	}

	public double getPrivacyExposure() {
		return this.privacyExposure;
	}

	/** Raw (non-normalized) interpretation cost of the selected items. */
	public double getInterpretationCost() {
		return this.interpretationCost;
	}

	public double getObjective() {
		return this.objective;
	}

	public boolean isSufficient() {
		return this.sufficient;
	}

	public boolean isPrivacyAdmissible() {
		return this.privacyAdmissible;
	}

	public boolean isMinimal() {
		return this.minimal;
	}
}
