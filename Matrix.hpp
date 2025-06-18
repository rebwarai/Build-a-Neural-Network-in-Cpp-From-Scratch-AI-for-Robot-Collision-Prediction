/*
author : @rebwar_ai
*/
#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>

class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows, cols;

public:
    // Matrix() : rows(0), cols(0), data() {}
    Matrix(size_t r, size_t c)
        : rows(r), cols(c) {
        data.resize(r, std::vector<double>(c, 0.0));
    }

    //Matrix Shape Methods ---->>
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }


    // Matrix operators Methods ---->>
    double& operator()(size_t i, size_t j) {
        if (i >= getRows() || j >= getCols()) 
        {
            throw std::out_of_range("Index out of bounds");
        }
            
        return data[i][j];
    }

    const double& operator()(size_t i, size_t j) const {
        if (i >= getRows() || j >= getCols()) 
        {
            throw std::out_of_range("Index out of bounds");
        }
        return data[i][j];
    }

    Matrix& fillRandom(double min = -1.0, double max = 1.0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(min, max);

        for (size_t i = 0; i < (*this).getRows(); ++i) {
            for (size_t j = 0; j < (*this).getCols(); ++j) {
                (*this)(i,j) = dist(gen);
            }
        }
        return *this;
    }

    Matrix operator*(double scalar) const {
        Matrix result(getRows(), getCols());
        for (size_t i = 0; i < getRows(); ++i)
        {
            for (size_t j = 0; j < getCols(); ++j)
            {
                result(i, j) = (*this)(i, j) * scalar;
            }
        }
        return result;
    }
    Matrix& operator*=(double scalar) {
        for (size_t i = 0; i < getRows(); ++i) {
            for (size_t j = 0; j < getCols(); ++j) {
                (*this)(i, j) *= scalar;
            }
        }
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix& mat) {
        for (size_t i = 0; i < mat.getRows(); ++i) 
        {
            for (size_t j = 0; j < mat.getCols(); ++j) 
            {
                os << mat(i, j) << " ";
            }
            os << "\n";
        }
        return os;
    }
};

#endif // MATRIX_H
