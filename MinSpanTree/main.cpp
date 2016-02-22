#include <iostream>
#include <set>
#include <vector>
#include <ctime>
#include <random>
#include <cmath>
#include <thread>
#include <cassert>
#include <fstream>
#include <sstream>
#include <ctime>
#include <String>

using namespace std;
const auto time_seed = static_cast<size_t>(time(0));
const auto clock_seed = static_cast<size_t>(clock());
const size_t pid_seed = hash<thread::id>()(this_thread::get_id());
seed_seq seed_val { time_seed, clock_seed, pid_seed };
mt19937_64 rand_gen;


/*
 GRAPH GENERATION
 */
typedef struct Vertex {

    vector<double> coords;
    Vertex* parent;
    int rank;
} Vertex;

typedef struct Edge {

    Vertex* u;
    Vertex* v;
    double distance;
} Edge;

typedef struct Graph {
    unsigned V;
    unsigned E;
    vector<Edge*> edges;
    vector<Vertex*> vertices;
} Graph;

// Used to generate points in space or edge weights in the '0D' graph
double generateRandomVal() {

    // Random value between 0 and 1
    return generate_canonical<double, 50>(rand_gen);
}

// Vertices == singleton sets in disjoint set data structure
Vertex* initializeVertex(vector<double> coords = {0}) {

    // Initialize vertex, make self the parent, and set rank to one
    Vertex* vertex = new Vertex();
    vertex->parent = vertex; //
    vertex->rank = 1;
    vertex->coords = coords;

    return vertex;
}

// Used in the 2D, 3D and 4D graphs
Vertex* generateRandomVertex(int dimension) {

    vector<double> coords(dimension);

    for (int i = 0; i < dimension; i++) {
        coords[i] = (generateRandomVal());
    }

    Vertex* vertex = initializeVertex(coords);

    return vertex;
}

/*
 The value of *d holds euclidean distance in the case in which we do not throw out the edge.
 It allows us to stop evaluation of distance without evaluating every pair of coordinates if a
 threshold is reached.
 */
bool calcEuclideanDist(Vertex* u, Vertex* v, double *d, double threshold) {
    double total = 0;
    for (int i = 0; i < (u->coords).size(); i++){
        total += pow((u->coords[i] - v->coords[i]),2);
        if (total > (threshold * threshold))
            return false;
    }
    *d = sqrt(total);
    return true;
}

/*
 See writeup for explanation of k(n) and error bounds. All residuals are based on the difference
 between approximated and observed data.
 */
double assignResidual(bool use_larger_residual, double low_n_bound, double high_n_bound) {
    if (use_larger_residual) {
        return low_n_bound;
    }
    return high_n_bound;
}

double calculatePruningThreshold(long n, int dimension){

    bool use_larger_residual = (n <= 1024);

    switch (dimension)
    {
        case 0:
            return 4.1218 * pow((double) n, -0.833) + use_larger_residual ? 0.06 0.3;

        case 2:
            return 2.0025 * pow((double) n, -0.459) + 0.08

        case 3:
            return 1.6973 * pow((double) n, -0.312) + use_larger_residual ? 0.2 : 0.05;

        case 4:
            return 1.7323 * pow((double) n, -0.249) + use_larger_residual ? 0.15 : 0.065;

        default:
            return sqrt(dimension);

    }

}

Graph generateGraph(unsigned size, int dimension, double weight_thresh) {

    vector<Vertex*> vertices(size);
    vector<Edge*> edges;

    bool in_euclidean_space = (dimension != 0);

    if (in_euclidean_space) {

        // Graph is in 2D, 3D or 4D, and coordinates in space matter
        for (int i = 0; i < size; i++) {
            vertices[i] = generateRandomVertex(dimension);
        }

    } else {

        /*
         Random egde weights: '0D' Case. Explicit vertices not needed. However, semantically
         it makes sense to have vertices regardless (it is a graph, after all). (Plus, we use vertices
         as singletons in the disjoint sets data structure for Kruskal's.)
         */
        for (int i = 0; i < size; i++) {

            // Vertices with arbitrary coordinates
            vertices[i] = initializeVertex();
        }
    }

    /*
     *
     * Vertices have been instantiated. Now for generating edge weights and pruning.
     *
     */

    for (int i = 0; i < size; i++) {

        for (int j = i + 1; j < size; j++) {

            Vertex* u = vertices[i];
            Vertex* v = vertices[j];

            // Get double pointer to hold total euclidean distance
            double* distance = new double();

            // Distance == Edge Weight
            if (in_euclidean_space) {

                // Dealing with a 2D, 3D or 4D graph. Must calculate distance and prune appropriately.
                if(calcEuclideanDist(u, v, distance, weight_thresh)){

                    /*
                     If euclidean distance is under threshold, then *distance is that value.
                     Otherwise, calcEuclideanDist returns false
                     */
                    Edge* new_edge = new Edge({u, v, *distance});
                    edges.push_back(new_edge);
                }

            } else {

                // All distances between 0 and 1. Coordinates in space wouldn't make sense here.
                *distance = generateRandomVal();

                // Throw out edges which are beyond the threshold
                if (*distance < weight_thresh) {
                    Edge* new_edge = new Edge({u, v, *distance});
                    edges.push_back(new_edge);
                }
            }

            free(distance);
        }
    }

    return (Graph){(unsigned) size, (unsigned) edges.size(), edges, vertices};
}


/*
 DISJOINT SET OPERATIONS
 */
Vertex* find(Vertex* v){
    if(v->parent != v)
        v->parent = find(v->parent);
    return v->parent;
}

void setUnion(Vertex* v, Vertex* u){
    Vertex* v_root = find(v);
    Vertex* u_root = find(u);
    if (v == u){
        return;
    }

    if (v_root->rank < u_root->rank){
        v_root->parent = u_root;
    }
    else if (v_root->rank > u_root->rank){
        u_root->parent = v_root;
    }
    else {
        u_root->parent = v_root;
        v_root->rank++;
    }

}


/*
 KRUSKAL'S MST ALGORITHM
 */
typedef struct MST {

    vector<Edge*> path;
    double total_weight;
} MST;

struct edgeCompare {
    bool operator() (Edge* e1, Edge* e2) {
        return (e1->distance < e2->distance);
    }
} edgeCompare;

void sortGraphEdgeList(Graph& G){
    sort(G.edges.begin(), G.edges.end(), edgeCompare);
}

MST findMST(Graph& G){

    MST foundMST;

    sortGraphEdgeList(G);
    for(Edge* E : G.edges){
        if (find(E->u) != find(E->v)){
            foundMST.path.push_back(E);
            //cout << "Added edge number " << foundMST.path.size() << "." << endl;
            foundMST.total_weight += E->distance;
            setUnion(E->u, E->v);
        }
    }
    return foundMST;
}


/*
 TESTING
 */

// Basic testing of Kruskal's implementation
void testHardcodedGraph() {

    // Hardcoded vertices and edges
    Vertex* A = initializeVertex();
    Vertex* B = initializeVertex();
    Vertex* C = initializeVertex();
    Vertex* D = initializeVertex();
    Vertex* E = initializeVertex();
    Vertex* F = initializeVertex();
    Vertex* G = initializeVertex();

    Edge* AB = new Edge({A, B, 7.0});
    Edge* AD = new Edge({A, D, 5.0});
    Edge* BC = new Edge({B, C, 8.0});
    Edge* BD = new Edge({B, D, 9.0});
    Edge* BE = new Edge({B, E, 7.0});
    Edge* CE = new Edge({C, E, 5.0});
    Edge* DE = new Edge({D, E, 15.0});
    Edge* DF = new Edge({D, F, 6.0});
    Edge* EF = new Edge({E, F, 7.0});
    Edge* EG = new Edge({E, G, 9.0});
    Edge* FG = new Edge({F, G, 11.0});

    // Hardcoded graph and true and false MSTs
    vector<Vertex*> vertices_list {A,B,C,D,E,F,G};
    vector<Edge*> edges_list {AB, AD, BC, BE, CE, EG, FG, DF, EF, DE, BD};
    Graph G_test {7, 11, edges_list, vertices_list};
    MST found_MST = findMST(G_test);

    vector<Edge*> true_path {AD, CE, DF, AB, BE, EG};
    double true_weight = 39;
    MST true_MST {true_path, true_weight};

    vector<Edge*> false_path {AD, CE, DF, AB, BE, EF};
    double false_weight = 40;
    MST false_MST {false_path, false_weight};

    assert(found_MST.path == true_MST.path);
    assert(found_MST.path != false_MST.path);

    for (Edge* E : G_test.edges)
        free(E);
    for (Vertex* V : G_test.vertices)
        free(V);

}

// Determines edge weight within the optimal MST. Used to determine k(n) as well as the error-bound for k(n)
void testMaxWeight(int dimension, string output_loc, int trials, int interval, int min_nodes, int max_nodes){
    ofstream outputFile(output_loc);

    for (int i = min_nodes; i <= max_nodes; i += interval){
        cout << "Doing " << trials << " trials for i = " << i << endl;
        double max = 0.0;
        for(int j = 0; j < trials; j++){

            // Absolutely ensures nothing is thrown out
            auto G = generateGraph(i, dimension, 100);
            auto MST = findMST(G);
            if(MST.path.back()->distance > max)
                max = MST.path.back()->distance;

            for (Edge* E : G.edges)
                free(E);
            for (Vertex* V : G.vertices)
                free(V);
        }
        outputFile << i << "\t" << max << endl;
        outputFile.close();
    }
}

void generateOutput() {
    rand_gen.seed(seed_val);
    //ofstream outputFile("OUTPUT.txt", ofstream::out);
    cout << "Size\t0\t\t2\t\3\t4\n";
    // loop through graph sizes
    for (int i = 16; i <= 65536; i *= 2) {
        //cout << "On graph size " << i << " for dimension " << endl;
        cout << i << "\t";
        // loop through dimensions
        for (int d = 0; d < 5; d++) {
            if (d == 1)
                continue;
            //cout << d << endl;
            // loop through trials
            double total = 0.0;
            for (int t = 0; t < 5; t++) {

                auto G = generateGraph(i, d, calculatePruningThreshold(i,d));

                auto MST = findMST(G);
                total += MST.total_weight;
                for (Edge* E : G.edges)
                    free(E);
                for (Vertex* V : G.vertices)
                    free(V);
                MST.total_weight = 0;
                MST.path.clear();
            }
            double avg_weight = total / 5;
            cout << avg_weight << "\t";

        }
        cout << "\n";
    }

}

// Ensures that total weight of the MST is similar when calculated with pruning and without pruning
void testPruning(int dimension, unsigned n) {

    rand_gen.seed(seed_val);

    auto G_p = generateGraph(n, dimension, calculatePruningThreshold(n, dimension));
    auto MST_p = findMST(G_p);

    cout << "Dimension " << dimension << " with pruning" << endl << "Length: " << MST_p.path.size() << " Total weight: " << MST_p.total_weight << endl;

    for (Edge* E : G_p.edges)
        free(E);
    for (Vertex* V : G_p.vertices)
        free(V);

    // 100 is an arbitrary upper bound
    auto G = generateGraph(n, dimension, 100);
    auto MST = findMST(G);

    cout << "Dimension " << dimension << " without pruning" << endl << "Length: " << MST.path.size() << " Total weight: " << MST.total_weight << endl;

    for (Edge* E : G.edges)
        free(E);
    for (Vertex* V : G.vertices)
        free(V);
}

void runCodeWithTiming(unsigned size, int trials, int dimension) {

    // Running code as CS 124 staff will with helpful output to console
    rand_gen.seed(seed_val);

    double total_search_time = 0;
    double avg_search_time = 0;

    for (int trial = 0; trial < trials; trial++) {

        clock_t gen_start_time = clock();
        auto G = generateGraph(size, dimension, calculatePruningThreshold(size, dimension));
        double gen_total_time = (clock() - gen_start_time) / (double)(CLOCKS_PER_SEC);

        cout << "Time for Graph Generation:    " << gen_total_time << "s" << endl;

        clock_t search_start_time = clock();
        auto MST = findMST(G);
        double search_total_time = (clock() - search_start_time) / (double)(CLOCKS_PER_SEC);

        cout << "Time for Trial " << trial + 1 << ":    " << search_total_time << "s" << endl;

        total_search_time += search_total_time;

        cout << "Lengh of path found: " << MST.path.size() << endl << "Total weight: " << MST.total_weight << endl;

        for (Edge* E : G.edges)
            free(E);
        for (Vertex* V : G.vertices)
            free(V);
    }

    avg_search_time = total_search_time / trials;

    cout << "Average search time over " << trials << " trials:    " << avg_search_time << "s" << endl;

}


/*
 COMMAND LINE INTERFACE

 Note that exit code of 0 is a success; 1 is an input failure; 2 is some other failure.
 */
int main(int argc, char** argv){

    if (argc != 5) {
        return 1;
    }

    vector<int> params;

    for (int i = 1; i < argc; i++) {

        istringstream char_param (argv[i]);
        int int_param;

        if (char_param >> int_param) {
            params.push_back(int_param);
        } else {
            return 1;
        }
    }

    int flag = params[0];
    unsigned size = params[1];
    int trials = params[2];
    int dimension = params[3];

    if (dimension == 1) {
        return 1;
    }

    if (flag == 0) {

        // TODO Output in the format requested in assignment
        rand_gen.seed(seed_val);
        double cumulative_weight = 0;

        for (int i = 0; i < trials; i++) {

            auto G = generateGraph(size, dimension, calculatePruningThreshold(size, dimension));
            auto MST = findMST(G);

            cumulative_weight += MST.total_weight;

            for (Edge* E : G.edges)
                free(E);
            for (Vertex* V : G.vertices)
                free(V);

        }

        double avg_weight = cumulative_weight / trials;

        cout << avg_weight << " " << size << " " << trials << " " << dimension << " " << endl;

        return 0;
    }

    if (flag == 1) {
        testHardcodedGraph();
        return 0;
    }

    if (flag == 2) {

        // Used to figure out k(n) and residuals. Dimension, output file name, numtrials, interval size, smallest n, largest n
        testMaxWeight(0, "0D.txt", 100, 5, 5, 400);
        return 0;
    }

    if (flag == 3) {

        // Uses same command line format as CS 124 tests
        runCodeWithTiming(size, trials, dimension);
        return 0;
    }

    if (flag == 4) {

        // First param is the dimension; second is the graph size
        testPruning(4, 8000);
    }

    if (flag == 5) {

        // Populate table in writeup
        generateOutput();
        return 0;
    }
}
