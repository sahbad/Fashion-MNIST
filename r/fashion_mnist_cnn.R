# Implements a 6-layer CNN using Keras in R, wrapped in an R6 class
# to classify Fashion MNIST images and predict sample items.

library(keras)
library(R6)

FashionMNISTClassifier <- R6Class(
  "FashionMNISTClassifier",
  public = list(
    x_train = NULL,
    y_train = NULL,
    x_test  = NULL,
    y_test  = NULL,
    model   = NULL,
    class_names = c(
      "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
      "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ),

    initialize = function() {
      # Load Fashion MNIST dataset
      dataset <- dataset_fashion_mnist()
      self$x_train <- dataset$train$x
      self$y_train <- dataset$train$y
      self$x_test  <- dataset$test$x
      self$y_test  <- dataset$test$y

      private$preprocess()
      self$model <- private$build_model()
    },

    train = function(epochs = 5, batch_size = 64) {
      history <- self$model %>% fit(
        x = self$x_train,
        y = self$y_train,
        epochs = epochs,
        batch_size = batch_size,
        validation_split = 0.1,
        verbose = 2
      )
      invisible(history)
    },

    evaluate = function() {
      results <- self$model %>% evaluate(self$x_test, self$y_test, verbose = 0)
      cat(sprintf("Test accuracy: %.4f\n", results["accuracy"]))
      invisible(results)
    },

      predict_samples = function(indices = c(1, 2)) {
      # Safer approach: flatten x_test to 2D, subset, then reshape back.

      x <- self$x_test
      d <- dim(x)
      cat("x_test dims:", paste(d, collapse = " x "), "\n")

      # d should be: batch_size x 28 x 28 x 1
      batch_size <- d[1]

      # 1) Flatten full test set to matrix: (batch_size, 28*28)
      flat <- array_reshape(x, c(batch_size, 28 * 28))

      # 2) Select only the requested indices as rows
      images_flat <- flat[indices, , drop = FALSE]

      # 3) Reshape back to (num_indices, 28, 28, 1)
      images <- array_reshape(
        images_flat,
        c(length(indices), 28, 28, 1)
      )

      # 4) Predict
      preds <- self$model %>% predict(images)
      predicted_classes <- apply(preds, 1, which.max) - 1  # convert to 0–9 labels

      # 5) Print nicely
      for (i in seq_along(indices)) {
        idx <- indices[i]
        true_label <- self$y_test[idx]
        pred_label <- predicted_classes[i]

        cat("Image index:", idx, "\n")
        cat("  True label:     ", self$class_names[true_label + 1], "\n")
        cat("  Predicted label:", self$class_names[pred_label + 1], "\n")
        cat(strrep("-", 40), "\n")
      }

      invisible(predicted_classes)
    }

  ),

  private = list(
    preprocess = function() {
      # Normalize to [0, 1]
      self$x_train <- self$x_train / 255
      self$x_test  <- self$x_test / 255

      # Ensure shape (batch, 28, 28, 1)
      self$x_train <- array_reshape(
        self$x_train,
        c(dim(self$x_train)[1], 28, 28, 1)
      )
      self$x_test <- array_reshape(
        self$x_test,
        c(dim(self$x_test)[1], 28, 28, 1)
      )
    },

    build_model = function() {
      model <- keras_model_sequential() %>%
        # 1
        layer_conv_2d(
          filters = 32, kernel_size = c(3, 3),
          activation = "relu",
          input_shape = c(28, 28, 1)
        ) %>%
        # 2
        layer_max_pooling_2d(pool_size = c(2, 2)) %>%
        # 3
        layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
        # 4
        layer_max_pooling_2d(pool_size = c(2, 2)) %>%
        # 5
        layer_flatten() %>%
        # 6
        layer_dense(units = 10, activation = "softmax")

      model %>% compile(
        optimizer = "adam",
        loss = "sparse_categorical_crossentropy",
        metrics = "accuracy"
      )
      model
    }
  )
)

# Run when sourced directly
if (sys.nframe() == 0) {
  cnn <- FashionMNISTClassifier$new()
  cnn$train(epochs = 5, batch_size = 64)
  cnn$evaluate()
  cnn$predict_samples(indices = c(1, 2))
}
