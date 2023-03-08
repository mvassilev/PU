// g++ -g -framework OpenCL mandelbrot_serial.cpp -o mandelbrot_serialcpp

#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

// Computes the Mandelbrot Set to N Iterations
void solve_mandelbrot(std::vector<float> const & real, std::vector<float> const & imag, int iterations, std::vector<int> & result) {
    for(unsigned int i = 0; i < real.size(); i++) {
        float x = real[i]; // Real Component
        float y = imag[i]; // Imaginary Component
        int   n = 0;       // Tracks Color Information

        // Compute the Mandelbrot Set
        while ((x * x + y * y <= 2 * 2) && n < iterations) {
            float xtemp = x * x - y * y + real[i];
            y = 2 * x * y + imag[i];
            x = xtemp;
            n++;
        }

        // Write Results to Output Arrays
        result[i] = x * x + y * y <= 2 * 2 ? -1 : n;
    }
}

// Draws a Portable Pixel Map (PPM/PBM) of the Given Mandelbrot Set
void ppm_draw(std::string const & filename, std::vector<std::vector<int> > const & grid) {
    // Declare Local Variables
    std::ofstream fd(filename);
    unsigned int iterations = 50;

    // Write the PPM Header
    fd << "P6"
       << " " << grid[0].size()
       << " " << grid.size()
       << " " << 255
       << std::endl;

    // Determine Pixel Color at Index
    for(auto &i : grid)
    for(auto &j : i)
    {
        unsigned char r, g, b;
        if(j == -1) { r = 0; g = 0; b = 0; }
        else
        {
            r = (j * 255 / iterations);
            g = r;
            b = 255;
        }

        // Write Colors to File
        fd << r << g << b;
    }
}

int main() {
    // Define Mandelbrot Settings
    int iterations = 2000;
    float x_min  = -2;
    float x_max  =  2;
    float y_min  = -1.5f;
    float y_max  =  1.5f;
    float x_step = .002f;
    float y_step = .002f;

    // Create Linear Vector of Coordinates
    std::vector<float> reals;
    std::vector<float> imags;
    for(float y = y_min; y < y_max; y += y_step)
    for(float x = x_min; x < x_max; x += x_step) {
        reals.push_back(x);
        imags.push_back(y);
    }

    // Compute Stride to Avoid Floating Point Errors
    unsigned int stride = 0;
    for (float x = x_min; x < x_max; x += x_step)
        stride++;

    // Compute the Mandelbrot Set on the CPU
    std::vector<int> cpu_ans(reals.size());
    clock_t cpu_begin = clock();

    solve_mandelbrot(reals, imags, iterations, cpu_ans);

    clock_t cpu_end = clock();
    clock_t cl_begin, cl_end;
 
    // Reshape the Output Array to 2D for Drawing
    std::vector<std::vector<int> > ans_grid;
    for(unsigned int i = 0; i < reals.size() / stride; i++)
        ans_grid.push_back(std::vector<int>(cpu_ans.begin() + (i * stride), cpu_ans.begin() + (i + 1) * stride));

    // Print Statistics
    std::cout << "CPU Time: " << (float) (cpu_end - cpu_begin) / CLOCKS_PER_SEC << "s" << std::endl;

    for (int i = 0; i < reals.size() ; i++) {
        printf("%3d, " , cpu_ans[i]);
    }

    // Write the Mandelbrot Set to File
    ppm_draw("mandelbrot2.ppm", ans_grid);

    return 0;
}