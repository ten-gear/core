import { expect, test } from "vitest";
import {
    chooseEnding,
    decodeExamples,
    encodeExamples,
    endingTargetMask,
    fitEnding,
    maskedMean,
    type HellaSwagExample,
} from "../project/hellaswag_data.js";
import { formatMetricRow, metricSteps, type MetricRow } from "../project/metrics.js";

test("HellaSwag records round-trip through the binary encoding", () => {
    const examples: HellaSwagExample[] = [
        {
            label: 3,
            context: Uint16Array.from([4, 5, 6]),
            endings: [
                Uint16Array.from([7]),
                Uint16Array.from([8, 9]),
                Uint16Array.from([10, 11, 12]),
                Uint16Array.from([13, 14]),
            ],
        },
        {
            label: 0,
            context: Uint16Array.from([]),
            endings: [
                Uint16Array.from([1, 2]),
                Uint16Array.from([3, 4, 5]),
                Uint16Array.from([6]),
                Uint16Array.from([7, 8, 9, 10]),
            ],
        },
    ];
    const decoded = decodeExamples(encodeExamples(examples));
    expect(decoded).toHaveLength(examples.length);
    for (let i = 0; i < examples.length; i++) {
        expect(decoded[i]!.label).toBe(examples[i]!.label);
        expect(Array.from(decoded[i]!.context)).toEqual(Array.from(examples[i]!.context));
        for (let ending = 0; ending < 4; ending++)
            expect(Array.from(decoded[i]!.endings[ending]!)).toEqual(Array.from(examples[i]!.endings[ending]!));
    }
});

test("the lower masked ending loss wins, and a tie keeps the earlier ending", () => {
    const contextHeavy = maskedMean([9, 1, 1], endingTargetMask(4, 2, 3));
    const endingHeavy = maskedMean([0, 4, 4], endingTargetMask(4, 2, 3));
    expect(contextHeavy).toBe(1);
    expect(endingHeavy).toBe(4);
    expect(chooseEnding([endingHeavy, contextHeavy, endingHeavy, endingHeavy])).toBe(1);
    expect(chooseEnding([2, 2, 2, 2])).toBe(0);
    expect(chooseEnding([3, 1, 1, 4])).toBe(1);

    const fitted = fitEnding(Uint16Array.from([1, 2, 3, 4, 5]), Uint16Array.from([6, 7, 8]), 6);
    expect(Array.from(fitted.tokens)).toEqual([3, 4, 5, 6, 7, 8]);
    expect(fitted.contextLength).toBe(3);
    expect(Array.from(endingTargetMask(fitted.tokens.length, fitted.contextLength, fitted.tokens.length - 1))).toEqual([0, 0, 1, 1, 1]);
});

test("metric rows include step 0, each interval, and a final step off the interval", () => {
    expect(metricSteps(250, 100)).toEqual([0, 100, 200, 250]);
    expect(metricSteps(200, 100)).toEqual([0, 100, 200]);

    const row: MetricRow = {
        step: 0,
        trainLoss: 10.5,
        valLoss: 10.25,
        learningRate: 0.0006,
        tokensSeen: 0,
        tokensPerSec: null,
        stepTimeMs: null,
        gradNorm: 1,
        weightNorm: 2,
        peakMemoryMb: 3,
        hellaswag: 0.25,
    };
    expect(formatMetricRow(row)).toBe("0,10.5,10.25,0.0006,0,,,1,2,3,0.25");
});
