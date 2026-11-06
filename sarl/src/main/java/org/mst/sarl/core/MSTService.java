package org.mst.sarl.core;

import java.util.ArrayList;
import java.util.List;

/**
 * Exact Minimum Sufficient Transparency (MST) service used by the role agents.
 *
 * <p>Each role and scenario has a three-item candidate pool. The service
 * enumerates all seven non-empty subsets, rejects any subset that contains an
 * inadmissible item, keeps the subsets whose utility reaches the threshold,
 * and returns the one with the lowest balanced objective
 * J = (P + C_norm + K) / 3. Objectives that differ by at most 1e-10 are broken
 * in favour of fewer items. If no admissible subset reaches the threshold, the
 * result is INFEASIBLE.</p>
 */
public final class MSTService {

	/** Tolerance used when comparing objective values. */
	public static final double TIE_TOLERANCE = 1e-10;

	private MSTService() {
	}

	/** One candidate information item. */
	public static final class Item {
		public final String id;
		public final double utility;
		public final double privacy;
		public final double cost;
		public final boolean admissible;

		public Item(String id, double utility, double privacy, double cost, boolean admissible) {
			this.id = id;
			this.utility = utility;
			this.privacy = privacy;
			this.cost = cost;
			this.admissible = admissible;
		}
	}

	/** Evaluates Exact-MST for one role, scenario and threshold. */
	public static MSTResult evaluate(String roleName, String scenario, double tau) {
		Item[] pool = candidatePool(roleName, scenario);
		int n = pool.length;
		double totalCost = 0.0;
		for (Item item : pool) {
			totalCost += item.cost;
		}

		int bestMask = -1;
		double bestObjective = Double.POSITIVE_INFINITY;
		int bestCount = Integer.MAX_VALUE;

		for (int mask = 1; mask < (1 << n); mask++) {
			if (!allAdmissible(pool, mask)) {
				continue;
			}
			if (utility(pool, mask) < tau) {
				continue;
			}
			double objective = objective(pool, mask, totalCost);
			int count = Integer.bitCount(mask);
			boolean better = objective < bestObjective - TIE_TOLERANCE;
			boolean tieWithFewerItems = Math.abs(objective - bestObjective) <= TIE_TOLERANCE && count < bestCount;
			if (better || tieWithFewerItems) {
				bestMask = mask;
				bestObjective = objective;
				bestCount = count;
			}
		}

		if (bestMask < 0) {
			return MSTResult.infeasible();
		}

		List<String> ids = new ArrayList<>();
		double rawCost = 0.0;
		for (int i = 0; i < n; i++) {
			if ((bestMask & (1 << i)) != 0) {
				ids.add(pool[i].id);
				rawCost += pool[i].cost;
			}
		}
		double u = utility(pool, bestMask);
		return new MSTResult(
				"FEASIBLE",
				ids,
				u,
				privacy(pool, bestMask),
				rawCost,
				bestObjective,
				u >= tau,
				allAdmissible(pool, bestMask),
				isDeletionMinimal(pool, bestMask, tau));
	}

	/**
	 * U(S) = 1 - prod(1 - u_i), computed incrementally as U <- U + u (1 - U),
	 * which is algebraically identical to the product form.
	 */
	static double utility(Item[] pool, int mask) {
		double u = 0.0;
		for (int i = 0; i < pool.length; i++) {
			if ((mask & (1 << i)) != 0) {
				u = u + pool[i].utility * (1.0 - u);
			}
		}
		return u;
	}

	/** P(S) = 1 - prod(1 - p_i). */
	static double privacy(Item[] pool, int mask) {
		double product = 1.0;
		for (int i = 0; i < pool.length; i++) {
			if ((mask & (1 << i)) != 0) {
				product *= (1.0 - pool[i].privacy);
			}
		}
		return 1.0 - product;
	}

	/** J(S) with cost normalized by the complete pool and K = |S| / n. */
	static double objective(Item[] pool, int mask, double totalCost) {
		double cost = 0.0;
		for (int i = 0; i < pool.length; i++) {
			if ((mask & (1 << i)) != 0) {
				cost += pool[i].cost;
			}
		}
		double normalizedCost = totalCost > 0.0 ? cost / totalCost : 0.0;
		double cardinality = (double) Integer.bitCount(mask) / pool.length;
		// Balanced weights: J = (P + C_norm + K) / 3.
		return (privacy(pool, mask) + normalizedCost + cardinality) / 3.0;
	}

	static boolean allAdmissible(Item[] pool, int mask) {
		for (int i = 0; i < pool.length; i++) {
			if ((mask & (1 << i)) != 0 && !pool[i].admissible) {
				return false;
			}
		}
		return true;
	}

	/** True when removing any single selected item breaks sufficiency. */
	static boolean isDeletionMinimal(Item[] pool, int mask, double tau) {
		for (int i = 0; i < pool.length; i++) {
			if ((mask & (1 << i)) != 0) {
				int reduced = mask & ~(1 << i);
				if (reduced != 0 && utility(pool, reduced) >= tau) {
					return false;
				}
			}
		}
		return true;
	}

	/** Controlled verification pools for the three scenarios and three roles. */
	public static Item[] candidatePool(String roleName, String scenario) {
		return switch (scenario) {
			case "HVAC_FAULT" ->
				hvacItems(roleName);
			case "COMFORT_DEVIATION" ->
				comfortItems(roleName);
			case "SECURITY_ALARM" ->
				securityItems(roleName);
			default -> throw new IllegalArgumentException("Unknown scenario: " + scenario);
		};
	}

	private static Item[] hvacItems(String roleName) {
		return switch (roleName) {
			case "OCCUPANT" -> new Item[] {
				new Item("zone_temperature", 0.70, 0.08, 0.20, true),
				new Item("comfort_deviation", 0.68, 0.05, 0.18, true),
				new Item("maintenance_staff_identity", 0.40, 0.45, 0.30, false),
			};
			case "FACILITY_MANAGER" -> new Item[] {
				new Item("hvac_fault_code", 0.82, 0.15, 0.30, true),
				new Item("airflow_anomaly", 0.58, 0.10, 0.24, true),
				new Item("zone_occupancy_count", 0.45, 0.25, 0.35, true),
			};
			case "SECURITY" -> new Item[] {
				new Item("safety_interlock_status", 0.76, 0.08, 0.20, true),
				new Item("affected_access_zone", 0.55, 0.16, 0.22, true),
				new Item("occupant_identity_trace", 0.42, 0.60, 0.40, false),
			};
			default -> throw new IllegalArgumentException("Unknown role: " + roleName);
		};
	}

	private static Item[] comfortItems(String roleName) {
		return switch (roleName) {
			case "OCCUPANT" -> new Item[] {
				new Item("zone_temperature", 0.72, 0.08, 0.18, true),
				new Item("comfort_target_difference", 0.64, 0.06, 0.18, true),
				new Item("neighbour_zone_occupancy", 0.42, 0.48, 0.34, false),
			};
			case "FACILITY_MANAGER" -> new Item[] {
				new Item("temperature_trend", 0.76, 0.10, 0.24, true),
				new Item("hvac_operating_mode", 0.66, 0.12, 0.26, true),
				new Item("aggregated_occupancy_level", 0.48, 0.22, 0.30, true),
			};
			case "SECURITY" -> new Item[] {
				new Item("safety_threshold_status", 0.78, 0.07, 0.18, true),
				new Item("affected_zone", 0.44, 0.12, 0.18, true),
				new Item("occupant_identity", 0.35, 0.58, 0.36, false),
			};
			default -> throw new IllegalArgumentException("Unknown role: " + roleName);
		};
	}

	private static Item[] securityItems(String roleName) {
		return switch (roleName) {
			case "OCCUPANT" -> new Item[] {
				new Item("alarm_status", 0.74, 0.07, 0.18, true),
				new Item("affected_access_zone", 0.58, 0.12, 0.20, true),
				new Item("camera_identity_trace", 0.46, 0.62, 0.40, false),
			};
			case "FACILITY_MANAGER" -> new Item[] {
				new Item("alarm_status", 0.72, 0.08, 0.18, true),
				new Item("affected_building_zone", 0.60, 0.14, 0.22, true),
				new Item("device_health_summary", 0.52, 0.12, 0.24, true),
			};
			case "SECURITY" -> new Item[] {
				new Item("alarm_status", 0.88, 0.10, 0.18, true),
				new Item("affected_access_zone", 0.62, 0.16, 0.20, true),
				new Item("camera_event_summary", 0.54, 0.24, 0.26, true),
			};
			default -> throw new IllegalArgumentException("Unknown role: " + roleName);
		};
	}
}
