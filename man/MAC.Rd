% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/Structure_improvement.R
\name{MAC}
\alias{MAC}
\title{MAC statistic.}
\usage{
MAC(X, Y, X2, Y2, k = NULL)
}
\arguments{
\item{X}{X is the predictors in the first data set and can be an n times p dimensional matrix, where n is the number of the points.}

\item{Y}{Y is the response in the first data set. It can be a vector of length n or an n times 1 dimensional matrix.}

\item{X2}{X2 is the predictors in the second data set and can be an n times p dimensional matrix.}

\item{Y2}{Y2 is the response in the second data set. The data type of Y2 should be consistent with Y.}

\item{k}{To speed up the calculation, when n is too large, you can specify to randomly extract k data points to calculate the MAC statistic.}
}
\value{
The MAC statistic between two (X, Y) and (X2, Y2).
}
\description{
Compute the MAC statistic between two data sets (X, Y) and (X2, Y2).
}
\examples{
   X <- matrix(rnorm(100), 10, 10)
   Y <- rnorm(10, 1, 1)
   X2 <- X
   Y2 <- rnorm(10, 2, 2)
   MAC(X, Y, X2, Y2)

}
