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
Vertex* generateRandomVertex(int dimensions) {

    vector<double> coords(dimensions);

    for (int i = 0; i < dimensions; i++) {
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
 See writeup for explanation of k(n) and error bound of 0.02, which was determined to be
 greater than all residuals found during the testing of k(n).
 */
double inline calculatePruningThreshold(long n, int dimension=3){

    switch (dimension)
    {
        case 0:
            return 1.1473 * pow((double) n, 0.296) + 0.02;

        case 2:
            return 2.1737 * pow((double) n, -0.474) + 0.02;

        case 3:
            return 1.6565 * pow((double) n, -0.306 ) + 0.02;

        case 4:
            return 1.141 * pow((double) n, -0.219) + 0.02;

        default:
            return 1;

    }

}

Graph generateGraph(unsigned size, int dimensions, double weight_thresh) {

    vector<Vertex*> vertices(size);
    vector<Edge*> edges;

    bool in_euclidean_space = (dimensions != 0);

    if (in_euclidean_space) {

        // Graph is in 2D, 3D or 4D, and coordinates in space matter
        for (int i = 0; i < size; i++) {
            vertices[i] = generateRandomVertex(dimensions);
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

void inline sortGraphEdgeList(Graph& G){
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

    double false_weight = 40;
    vector<Edge*> false_path {AD, CE, DF, AB, BE, EF};
    MST false_MST {false_path, false_weight};

    // Test
    assert(found_MST.path == true_MST.path && found_MST.total_weight == true_MST.total_weight);
    assert(found_MST.path != false_MST.path && found_MST.total_weight != false_MST.total_weight);

    for (Edge* E : G_test.edges)
        free(E);
    for (Vertex* V : G_test.vertices)
        free(V);

}

void testMaxWeight(int dimensions, string outputLoc, int numTrials, int minNodes, int maxNodes){
    rand_gen.seed(seed_val);
    ofstream outputFile(outputLoc);
    for (int i = minNodes; i <= maxNodes; i += 5){
        cout << "Doing " << numTrials << " trials for i = " << i << endl;
        double max = 0.0;
        for(int j = 0; j < numTrials; j++){
            auto G = generateGraph(i, dimensions, 1.0);
            auto MST = findMST(G);
            if(MST.path.back()->distance > max)
                max = MST.path.back()->distance;

            // deallocate pointers
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
                
                auto G = generateGraph(i, d, calculatePruningThreshold(i));
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

/*
 COMMAND LINE INTERFACE

 Note that exit code of 0 is a success; 1 is an input failure; 2 is some other failure.
 */
int main(int argc, char** argv){
    // TODO add double threshold parameter

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
    int dimensions = params[3];

    if (dimensions == 1) {
        return 1;
    }
    
    rand_gen.seed(seed_val);
    
    if (flag == 1) {
        cout << "\nTesting\n";
        testHardcodedGraph();
        cout << "\nMST Working on Hardcoded Graph\n";
        cout << "\nAll Tests Pass\n";

        return 0;
    }

    if (flag == 2) {
        testMaxWeight(2, "5_400_100_trials_2D.txt", 100, 5, 400);
        return 0;
    }
    if (flag == 3) {
        generateOutput();
        return 0;
    }

    double total_search_time = 0;
    double avg_search_time = 0;

    for (int trial = 0; trial < trials; trial++) {

        rand_gen.seed(seed_val);

        clock_t gen_start_time = clock();
        auto G = generateGraph(size, dimensions, calculatePruningThreshold(size, dimensions));
        double gen_total_time = (clock() - gen_start_time) / (double)(CLOCKS_PER_SEC);

        cout << "Time for Graph Generation:    " << gen_total_time << "s" << endl;

        clock_t search_start_time = clock();
        auto MST = findMST(G);
        double search_total_time = (clock() - search_start_time) / (double)(CLOCKS_PER_SEC);

        cout << "Time for Trial " << trial + 1 << ":    " << search_total_time << "s" << endl;

        total_search_time += search_total_time;

        for (Edge* E : G.edges)
            free(E);
        for (Vertex* V : G.vertices)
            free(V);
    }

    avg_search_time = total_search_time / trials;

    cout << "Average search time over " << trials << " trials:    " << avg_search_time << "s" << endl;

    return 0;
}
