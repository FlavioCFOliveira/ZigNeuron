/// Evaluation metrics for classification and regression
///
/// Classification Metrics:
/// - Accuracy: (TP + TN) / (TP + TN + FP + FN)
/// - Precision: TP / (TP + FP)
/// - Recall: TP / (TP + FN)
/// - F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
/// - Confusion Matrix
///
/// Regression Metrics:
/// - MAE (Mean Absolute Error)
/// - MSE (Mean Squared Error)
/// - RMSE (Root Mean Squared Error)
///
/// References:
/// - Accuracy, Precision, Recall: Powers, D. M. W. (2011). Evaluation: From precision,
///   recall and F-measure to ROC, informedness, markedness and correlation. Journal of
///   Machine Learning Technologies, 2(1), 37-63.
const std = @import("std");

/// Classification metrics computed from predictions and targets
pub const ClassificationMetrics = struct {
    accuracy: f32,
    precision: f32,
    recall: f32,
    f1_score: f32,
    confusion_matrix: [][]usize,

    pub fn deinit(self: *ClassificationMetrics, allocator: std.mem.Allocator) void {
        for (self.confusion_matrix) |row| {
            allocator.free(row);
        }
        allocator.free(self.confusion_matrix);
    }

    pub fn print(self: ClassificationMetrics) void {
        std.debug.print("\n=== Classification Metrics ===\n", .{});
        std.debug.print("Accuracy:  {d:.4}\n", .{self.accuracy});
        std.debug.print("Precision: {d:.4}\n", .{self.precision});
        std.debug.print("Recall:    {d:.4}\n", .{self.recall});
        std.debug.print("F1-Score:  {d:.4}\n", .{self.f1_score});

        std.debug.print("\nConfusion Matrix:\n", .{});
        const num_classes = self.confusion_matrix.len;
        // Header
        std.debug.print("      ", .{});
        for (0..num_classes) |c| {
            std.debug.print("Pred{d:<3}", .{c});
        }
        std.debug.print("\n", .{});

        // Rows
        for (self.confusion_matrix, 0..) |row, i| {
            std.debug.print("True{d:<2} ", .{i});
            for (row) |val| {
                std.debug.print("{d:>5} ", .{val});
            }
            std.debug.print("\n", .{});
        }
    }
};

/// Regression metrics
pub const RegressionMetrics = struct {
    mae: f32, // Mean Absolute Error
    mse: f32, // Mean Squared Error
    rmse: f32, // Root Mean Squared Error
    mape: f32, // Mean Absolute Percentage Error
    r_squared: f32, // Coefficient of determination

    pub fn print(self: RegressionMetrics) void {
        std.debug.print("\n=== Regression Metrics ===\n", .{});
        std.debug.print("MAE:        {d:.6}\n", .{self.mae});
        std.debug.print("MSE:        {d:.6}\n", .{self.mse});
        std.debug.print("RMSE:       {d:.6}\n", .{self.rmse});
        std.debug.print("MAPE:       {d:.4}%\n", .{self.mape * 100});
        std.debug.print("R-squared:  {d:.6}\n", .{self.r_squared});
    }
};

/// Compute classification metrics
/// predictions: array of predicted class indices
/// targets: array of true class indices
/// num_classes: number of classes
pub fn computeClassificationMetrics(
    allocator: std.mem.Allocator,
    predictions: []const usize,
    targets: []const usize,
    num_classes: usize,
) !ClassificationMetrics {
    if (predictions.len != targets.len) return error.SizeMismatch;
    if (predictions.len == 0) return error.EmptyInput;

    // Allocate confusion matrix
    var cm = try allocator.alloc([]usize, num_classes);
    errdefer allocator.free(cm);

    for (0..num_classes) |i| {
        cm[i] = try allocator.alloc(usize, num_classes);
        @memset(cm[i], 0);
    }

    // Fill confusion matrix
    for (predictions, targets) |pred, target| {
        if (pred >= num_classes or target >= num_classes) {
            // Clean up on error
            for (cm) |row| allocator.free(row);
            allocator.free(cm);
            return error.InvalidClass;
        }
        cm[target][pred] += 1;
    }

    // Compute metrics
    var total_correct: usize = 0;
    const total: usize = predictions.len;
    var total_precision: f32 = 0;
    var total_recall: f32 = 0;

    for (0..num_classes) |c| {
        const tp: usize = cm[c][c];
        var fp: usize = 0;
        var fn_count: usize = 0;

        for (0..num_classes) |i| {
            if (i != c) {
                fp += cm[i][c]; // Predicted c but was i
                fn_count += cm[c][i]; // Was c but predicted i
            }
        }

        total_correct += tp;

        // Per-class precision and recall
        const class_precision = if (tp + fp > 0)
            @as(f32, @floatFromInt(tp)) / @as(f32, @floatFromInt(tp + fp))
        else
            0;
        const class_recall = if (tp + fn_count > 0)
            @as(f32, @floatFromInt(tp)) / @as(f32, @floatFromInt(tp + fn_count))
        else
            0;

        total_precision += class_precision;
        total_recall += class_recall;
    }

    const accuracy = @as(f32, @floatFromInt(total_correct)) / @as(f32, @floatFromInt(total));
    const precision = total_precision / @as(f32, @floatFromInt(num_classes));
    const recall = total_recall / @as(f32, @floatFromInt(num_classes));
    const f1_score = if (precision + recall > 0)
        2.0 * precision * recall / (precision + recall)
    else
        0;

    return ClassificationMetrics{
        .accuracy = accuracy,
        .precision = precision,
        .recall = recall,
        .f1_score = f1_score,
        .confusion_matrix = cm,
    };
}

/// Compute regression metrics
pub fn computeRegressionMetrics(
    predictions: []const f32,
    targets: []const f32,
) !RegressionMetrics {
    if (predictions.len != targets.len) return error.SizeMismatch;
    if (predictions.len == 0) return error.EmptyInput;

    const n = predictions.len;
    const n_f = @as(f32, @floatFromInt(n));

    var mae_sum: f32 = 0;
    var mse_sum: f32 = 0;
    var mape_sum: f32 = 0;

    // For R-squared
    var mean_target: f32 = 0;
    for (targets) |t| {
        mean_target += t;
    }
    mean_target /= n_f;

    var ss_tot: f32 = 0; // Total sum of squares
    var ss_res: f32 = 0; // Residual sum of squares

    for (predictions, targets) |pred, target| {
        const diff = pred - target;
        const abs_diff = @abs(diff);

        mae_sum += abs_diff;
        mse_sum += diff * diff;
        mape_sum += abs_diff / @max(@abs(target), 1e-8); // Avoid division by zero

        ss_tot += (target - mean_target) * (target - mean_target);
        ss_res += diff * diff;
    }

    const mae = mae_sum / n_f;
    const mse = mse_sum / n_f;
    const rmse = @sqrt(mse);
    const mape = mape_sum / n_f;
    const r_squared = if (ss_tot > 0) 1.0 - (ss_res / ss_tot) else 0;

    return RegressionMetrics{
        .mae = mae,
        .mse = mse,
        .rmse = rmse,
        .mape = mape,
        .r_squared = r_squared,
    };
}

/// Get predicted class from one-hot encoded output
pub fn getPredictedClass(output: []const f32) usize {
    var max_idx: usize = 0;
    var max_val = output[0];
    for (output[1..], 1..) |val, i| {
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    return max_idx;
}

/// Get true class from one-hot encoded target
pub fn getTrueClass(target: []const f32) usize {
    for (target, 0..) |t, i| {
        if (t > 0.5) return i;
    }
    return 0;
}

/// Evaluate classification model
/// Returns metrics for model evaluation
pub fn evaluateClassification(
    allocator: std.mem.Allocator,
    outputs: []const []const f32,
    targets: []const []const f32,
    num_classes: usize,
) !ClassificationMetrics {
    if (outputs.len != targets.len) return error.SizeMismatch;

    // Convert outputs and targets to class indices
    var predictions = try allocator.alloc(usize, outputs.len);
    defer allocator.free(predictions);
    var class_targets = try allocator.alloc(usize, targets.len);
    defer allocator.free(class_targets);

    for (outputs, 0..) |output, i| {
        predictions[i] = getPredictedClass(output);
    }

    for (targets, 0..) |target, i| {
        class_targets[i] = getTrueClass(target);
    }

    return try computeClassificationMetrics(allocator, predictions, class_targets, num_classes);
}

/// Evaluate regression model
pub fn evaluateRegression(
    outputs: []const []const f32,
    targets: []const []const f32,
) !RegressionMetrics {
    if (outputs.len != targets.len) return error.SizeMismatch;

    // Flatten outputs and targets
    const n = outputs.len;
    const dim = outputs[0].len;

    var pred_flat = try std.heap.page_allocator.alloc(f32, n * dim);
    defer std.heap.page_allocator.free(pred_flat);
    var target_flat = try std.heap.page_allocator.alloc(f32, n * dim);
    defer std.heap.page_allocator.free(target_flat);

    for (outputs, 0..) |output, i| {
        for (output, 0..) |val, j| {
            pred_flat[i * dim + j] = val;
        }
    }

    for (targets, 0..) |target, i| {
        for (target, 0..) |val, j| {
            target_flat[i * dim + j] = val;
        }
    }

    return try computeRegressionMetrics(pred_flat, target_flat);
}

/// Print per-class metrics
pub fn printPerClassMetrics(
    cm: []const []const usize,
) void {
    const num_classes = cm.len;

    std.debug.print("\n=== Per-Class Metrics ===\n", .{});
    std.debug.print("{s:<8} {s:>10} {s:>10} {s:>10} {s:>10}\n", .{ "Class", "Precision", "Recall", "F1-Score", "Support" });

    var total_precision: f32 = 0;
    var total_recall: f32 = 0;
    var total_f1: f32 = 0;
    var valid_classes: usize = 0;

    for (0..num_classes) |c| {
        const tp = cm[c][c];
        var fp: usize = 0;
        var fn_count: usize = 0;
        var support: usize = 0;

        for (0..num_classes) |i| {
            support += cm[c][i];
            if (i != c) {
                fp += cm[i][c];
                fn_count += cm[c][i];
            }
        }

        const precision = if (tp + fp > 0)
            @as(f32, @floatFromInt(tp)) / @as(f32, @floatFromInt(tp + fp))
        else
            0;
        const recall = if (tp + fn_count > 0)
            @as(f32, @floatFromInt(tp)) / @as(f32, @floatFromInt(tp + fn_count))
        else
            0;
        const f1 = if (precision + recall > 0)
            2.0 * precision * recall / (precision + recall)
        else
            0;

        total_precision += precision;
        total_recall += recall;
        total_f1 += f1;
        valid_classes += 1;

        std.debug.print("{d:<8} {d:>10.4} {d:>10.4} {d:>10.4} {d:>10}\n", .{ c, precision, recall, f1, support });
    }

    // Macro average
    if (valid_classes > 0) {
        std.debug.print("{s:<8} {d:>10.4} {d:>10.4} {d:>10.4}\n", .{
            "macro",
            total_precision / @as(f32, @floatFromInt(valid_classes)),
            total_recall / @as(f32, @floatFromInt(valid_classes)),
            total_f1 / @as(f32, @floatFromInt(valid_classes)),
        });
    }
}
