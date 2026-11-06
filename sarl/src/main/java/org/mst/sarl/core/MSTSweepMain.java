package org.mst.sarl.core;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Plain-Java execution of the 36-condition sweep, without the SARL agents.
 *
 * <p>It calls the same {@link MSTService} and {@link MSTSweepRecorder} as the
 * agents, in a fixed order, and is used to check the service and recorder
 * independently of agent scheduling.</p>
 */
public final class MSTSweepMain {

	public static final String[] SCENARIOS = {"HVAC_FAULT", "COMFORT_DEVIATION", "SECURITY_ALARM"};
	public static final String[] ROLES = {"OCCUPANT", "FACILITY_MANAGER", "SECURITY"};
	public static final double[] THRESHOLDS = {0.70, 0.80, 0.90, 0.95};

	private MSTSweepMain() {
	}

	public static void main(String[] args) {
		Path output = args.length > 0 ? Paths.get(args[0]) : Paths.get("results", MSTSweepRecorder.FILE_NAME);
		MSTSweepRecorder recorder = new MSTSweepRecorder(output);
		recorder.open();
		for (String scenario : SCENARIOS) {
			for (double tau : THRESHOLDS) {
				for (String role : ROLES) {
					MSTResult r = MSTService.evaluate(role, scenario, tau);
					recorder.record(scenario, tau, role, r.getStatus(), r.getSelectedItems(),
							r.getUtility(), r.getPrivacyExposure(), r.getInterpretationCost(),
							r.getObjective(), r.isSufficient(), r.isPrivacyAdmissible(), r.isMinimal());
				}
			}
		}
		recorder.close();
		System.out.println("MULTI-SCENARIO SWEEP COMPLETE | conditions=" + recorder.getRows()
				+ " | verified=" + recorder.getVerified() + "/" + recorder.getRows()
				+ " | nested=" + recorder.getNestedPassed() + "/" + recorder.getNestedApplicable()
				+ " | output=" + recorder.getOutputFile());
	}
}
