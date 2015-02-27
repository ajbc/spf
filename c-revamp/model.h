class Model {
    protected:
        Data* data;
    
    public:
        virtual double predict(int user, int item) { return 0; };
};
