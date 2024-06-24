#include <stdlib.h>
#include <math.h>

typedef struct 
{
    float* data;
    int* shape;
    int n_dims;
    int size;
} Array;

void cleanup_array(Array* arr)
{
    free(arr->data);
    free(arr->shape);
    free(arr);
}

Array* create_array(int* shape, int n_dims)
{
    Array* arr = (Array*)malloc(sizeof(Array));
    arr->n_dims = n_dims;
    arr->shape = (int*)malloc(n_dims * sizeof(int));
    arr->size = 1;
    for (int i = 0; i < n_dims; i++)
    {
        arr->shape[i] = shape[i];
        arr->size *= shape[i];
    }

    arr->data = (float*)malloc(arr->size * sizeof(float));
    return arr;
}

void fill_array(Array* arr, float value)
{
    for (int i = 0; i < arr->size; i++)
    {
        arr->data[i] = value;
    }
}

Array* sum_arrays(Array* arr1, Array* arr2)
{
    // Same shape is required!! Check this in the calling code

    Array* result = create_array(arr1->shape, arr1->n_dims);
    for (int i = 0; i < arr1->size; i++)
    {
        result->data[i] = arr1->data[i] + arr2->data[i];
    }

    return result;
}

Array* sum_scalar(Array* arr, float scalar)
{
    Array* result = create_array(arr->shape, arr->n_dims);
    for (int i = 0; i < arr->size; i++)
    {
        result->data[i] = arr->data[i] + scalar;
    }

    return result;
}

Array* prod_scalar(Array* arr, float scalar)
{
    Array* result = create_array(arr->shape, arr->n_dims);
    for (int i = 0; i < arr->size; i++)
    {
        result->data[i] = arr->data[i] * scalar;
    }

    return result;
}

Array* pow_scalar(Array* arr, float power)
{
    Array* result = create_array(arr->shape, arr->n_dims);
    for (int i = 0; i < arr->size; i++)
    {
        result->data[i] = powf(arr->data[i], power);
    }

    return result;
}

Array* exp_array(Array* arr)
{
    Array* result = create_array(arr->shape, arr->n_dims);
    for (int i = 0; i < arr->size; i++)
    {
        result->data[i] = expf(arr->data[i]);
    }

    return result;
}

Array* multiply_matrix(Array* arr1, Array* arr2)
{
    // Check if the shapes are compatible
    if (arr1->n_dims != 2 || arr2->n_dims != 2)
    {
        return NULL;
    }
    if (arr1->shape[1] != arr2->shape[0])
    {
        return NULL;
    }

    Array* result = create_array((int[]){arr1->shape[0], arr2->shape[1]}, 2);
    for (int i = 0; i < arr1->shape[0]; i++)
    {
        for (int j = 0; j < arr2->shape[1]; j++)
        {
            float sum = 0;
            for (int k = 0; k < arr1->shape[1]; k++)
            {
                sum += arr1->data[i * arr1->shape[1] + k] * arr2->data[k * arr2->shape[1] + j];
            }
            result->data[i * result->shape[1] + j] = sum;
        }
    }

    return result;
}

Array* multiply_vector(Array* arr1, Array* arr2)
{
    // Check if the shapes are compatible
    if (arr1->n_dims != 2 || arr2->n_dims != 1)
    {
        return NULL;
    }
    if (arr1->shape[1] != arr2->shape[0])
    {
        return NULL;
    }

    Array* result = create_array((int[]){arr1->shape[0]}, 1);
    for(int i = 0; i < arr1->shape[0]; i++)
    {
        float sum = 0;
        for (int j = 0; j < arr1->shape[1]; j++)
        {
            sum += arr1->data[i * arr1->shape[1] + j] * arr2->data[j];
        }
        result->data[i] = sum;
    }


    return result;
}

float sum_array_elements(Array* arr)
{
    float sum = 0;
    for (int i = 0; i < arr->size; i++)
    {
        sum += arr->data[i];
    }

    return sum;
}

Array* stack_arrays(Array** arrs, int n_arrs)
{
    // Check if the shapes are compatible
    int n_dims = arrs[0]->n_dims;
    for (int i = 1; i < n_arrs; i++)
    {
        if (arrs[i]->n_dims != n_dims)
        {
            return NULL;
        }
        for (int j = 0; j < n_dims; j++)
        {
            if (arrs[i]->shape[j] != arrs[0]->shape[j])
            {
                return NULL;
            }
        }
    }

    int* new_shape = (int*)malloc((n_dims + 1) * sizeof(int));
    new_shape[n_dims] = n_arrs;
    for (int i = 0; i < n_dims; i++)
    {
        new_shape[i] = arrs[0]->shape[i];
    }

    Array* result = create_array(new_shape, n_dims + 1);
    for (int i = 0; i < n_arrs; i++)
    {
        for (int j = 0; j < arrs[i]->size; j++)
        {
            result->data[i * arrs[i]->size + j] = arrs[i]->data[j];
        }
    }

    return result;
}

Array* arange(int start, int end, int step)
{
    int n = (end - start) / step;
    Array* result = create_array((int[]){n}, 1);
    for (int i = 0; i < n; i++)
    {
        result->data[i] = (float)(start + i * step);
    }

    return result;
}

Array *create_rand(int *shape, int n_dims)
{
    Array *arr = create_array(shape, n_dims);
    for (int i = 0; i < arr->size; i++)
    {
        arr->data[i] = (float)rand() / RAND_MAX;
    }

    return arr;
}