#include "nbl/application_templates/BasicMultiQueueApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"

using namespace nbl;
using namespace core;
using namespace asset;
using namespace system;
using namespace video;

class ComputeScanApp final : public application_templates::BasicMultiQueueApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
    using device_base_t = application_templates::BasicMultiQueueApplication;
    using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;

public:
    ComputeScanApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
        system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

    bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
    {
        if (!device_base_t::onAppInitialized(std::move(system)))
            return false;
        if (!asset_base_t::onAppInitialized(std::move(system)))
            return false;

        return true;
    }

    virtual bool onAppTerminated() override
    {
        return true;
    }

    // the unit test is carried out on init
    void workLoopBody() override {}

    //
    bool keepRunning() override { return false; }

private:
    // nothing
};

NBL_MAIN_FUNC(ComputeScanApp)