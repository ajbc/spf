class Model {
    protected:
        Data* data;
    
    public:
        virtual double prediction(int user, int item) { return 0; };
};
