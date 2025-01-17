

#' Coverage ratio between training and test sets.
#' @description Coverage Ratio is a metric that measures the similarity between the training set and the test set.
#'     Based on the dimensional reduction graph of tsne, we calculated how many data points in the test set have k-nearest neighbors that contain the training set, that is,
#'     they are covered by the training set. If CR < 0.5, we believe that this test set is significantly different from the training set and may not be applicable to the corresponding model.
#'
#' @param X_train The predictors in the training set, an n-by-p matrix.
#' @param X_test The predictors in the training set, a matrix with the same number of columns as X.
#'
#' @returns The Coverage ratio between train and test sets.
#' @export
#'
#' @examples
#'     X_train <- matrix(rnorm(2000), 200, 10)
#'     X_test <- matrix(rnorm(1000, 1), 100, 10)
#'     coverage_ratio(X_train, X_test)
coverage_ratio <- function(X_train, X_test) {
  X_all <- as.matrix(rbind(X_train, X_test))
  tsne <- Rtsne::Rtsne(X_all, check_duplicates = FALSE)
  y <- tsne$Y[1:nrow(X_train), ]
  x <- tsne$Y[(nrow(X_train)+1):nrow(X_all), ]
  dist_x <- as.matrix(stats::dist(x))
  diag(dist_x) <- rep(100000, dim(x)[1])
  k <- round(0.02 * length(x)) + 1
  nearest_dist_x <- apply(dist_x, 1, function(xi) {sort(xi)[k]})
  nearest_dist_xy <- apply(x, 1, function(x_row) {
    min(sqrt(colSums((t(y) - x_row)^2)))
  })
  return(sum(nearest_dist_xy < nearest_dist_x) / length(nearest_dist_xy))
}
